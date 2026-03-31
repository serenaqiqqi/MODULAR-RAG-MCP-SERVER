"""File integrity checker for incremental ingestion.

This module provides SHA256-based file integrity tracking to enable incremental
ingestion. Files that have been successfully processed can be skipped on
subsequent ingestion runs.

Design Principles:
- Idempotent: Multiple ingestion runs of the same file are safe
- Persistent: SQLite-backed storage survives process restarts
- Concurrent: WAL mode enables concurrent read/write operations
- Graceful: Failed ingestions are tracked but don't block retries
"""
# 上面：用文件 SHA256 记「是否已成功入库」，下次跑 ingest 可跳过未改文件；库在 SQLite 里。

import hashlib  # 计算 SHA256
import sqlite3  # 本地数据库，存摄取历史 sqlite3 就是让你在本地用一个 .db 文件当数据库来存和查数据的工具
from abc import ABC, abstractmethod  # 抽象基类 + 抽象方法装饰器
from datetime import datetime, timezone  # UTC 时间戳字符串
from pathlib import Path  # 路径校验
from typing import Any, Dict, List, Optional  # 类型注解


class FileIntegrityChecker(ABC):  # 接口：不同后端可实现自己的 Checker
    """Abstract base class for file integrity checking.
    
    Implementations track which files have been successfully processed
    to enable incremental ingestion.
    """
    # 子类必须实现：算哈希、判断是否跳过、标记成功/失败、删记录、列成功列表

    @abstractmethod  # 子类必须实现下面这个方法
    def compute_sha256(self, file_path: str) -> str:  # 读文件算 SHA256
        """Compute SHA256 hash of file.
        
        Args:
            file_path: Path to the file to hash.
            
        Returns:
            Hexadecimal SHA256 hash string (64 characters).
            
        Raises:
            FileNotFoundError: If file does not exist.
            IOError: If path is not a file or cannot be read.
        """
        pass  # 由具体存储实现（如 SQLite）

    @abstractmethod
    def should_skip(self, file_hash: str) -> bool:  # 是否跳过本次 ingest
        """Check if file should be skipped based on hash.
        
        Args:
            file_hash: SHA256 hash of the file.
            
        Returns:
            True if file has been successfully processed before, False otherwise.
        """
        pass  # 子类查库：只有 success 才跳过

    @abstractmethod
    def mark_success(
        self,
        file_hash: str,
        file_path: str,
        collection: Optional[str] = None
    ) -> None:  # 入库成功后记一笔
        """Mark file as successfully processed.
        
        Args:
            file_hash: SHA256 hash of the file.
            file_path: Original file path (for tracking).
            collection: Optional collection/namespace identifier.
            
        Raises:
            RuntimeError: If database operation fails.
        """
        pass  # 子类写入「已成功」

    @abstractmethod
    def mark_failed(
        self,
        file_hash: str,
        file_path: str,
        error_msg: str
    ) -> None:  # 入库失败也记一笔，方便重试
        """Mark file processing as failed.
        
        Failed files are tracked but not skipped on subsequent runs,
        allowing retries.
        
        Args:
            file_hash: SHA256 hash of the file.
            file_path: Original file path (for tracking).
            error_msg: Error message describing the failure.
            
        Raises:
            RuntimeError: If database operation fails.
        """
        pass  # 子类写入「失败」并记错误信息，下次不跳过可重试

    @abstractmethod
    def remove_record(self, file_hash: str) -> bool:  # 删库里的历史记录
        """Remove an ingestion record by its file hash.

        Args:
            file_hash: SHA256 hash identifying the record.

        Returns:
            True if a record was deleted, False if not found.
        """
        pass  # 子类按哈希删一条历史（例如删文档后清记录）

    @abstractmethod
    def list_processed(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:  # 列出已成功处理的文件
        """List successfully processed files.

        Args:
            collection: Optional collection filter.  When *None* all
                successful records are returned.

        Returns:
            List of dicts with keys: file_hash, file_path, collection,
            processed_at, updated_at.
        """
        pass  # 子类列出所有 success 记录，可按 collection 过滤


class SQLiteIntegrityChecker(FileIntegrityChecker):  # 默认实现：SQLite 持久化
    """SQLite-backed file integrity checker.
    
    Stores ingestion history in a SQLite database with WAL mode for
    concurrent access.
    
    Database Schema:
        ingestion_history (
            file_hash TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            status TEXT NOT NULL,  -- 'success' or 'failed'
            collection TEXT,
            error_msg TEXT,
            processed_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        )
    
    Args:
        db_path: Path to SQLite database file (will be created if needed).
    
    Raises:
        sqlite3.DatabaseError: If database file is corrupted.
    """
    # 表 ingestion_history：主键 file_hash，status 区分 success/failed，WAL 方便并发读

    def __init__(self, db_path: str):
        """Initialize checker and create database if needed.
        
        Args:
            db_path: Path to SQLite database file.
        """
        self.db_path = db_path  # 数据库文件路径字符串
        self._conn = None  # 预留长连接位；当前实现每次操作短连接，多为 None
        self._ensure_database()  # 建库、建表、开 WAL

    def close(self) -> None:
        """Close database connection if open."""
        if self._conn:  # 若以后改用长连接，这里关掉
            self._conn.close()
            self._conn = None

    def __del__(self):
        """Cleanup: close connection on deletion."""
        self.close()  # 对象销毁时尽量释放连接

    #开工前先把数据库环境搭好
    def _ensure_database(self) -> None:
        """Create database file and schema if they don't exist."""
        # Create parent directories if needed
        db_file = Path(self.db_path)  # 例如 data/db/ingestion_history.db
        db_file.parent.mkdir(parents=True, exist_ok=True)  # 父目录不存在就创建

        # Connect and initialize schema
        conn = sqlite3.connect(self.db_path)  # 打开或创建 sqlite 文件，连上数据库，没有就新建一个
        try:
            # Enable WAL mode for concurrent access
            conn.execute("PRAGMA journal_mode=WAL")  # 打开 SQLite 的 WAL 模式，让数据库更适合并发读写

            # Create table if not exists
            conn.execute("""
                CREATE TABLE IF NOT EXISTS ingestion_history (
                    file_hash TEXT PRIMARY KEY,
                    file_path TEXT NOT NULL,
                    status TEXT NOT NULL,
                    collection TEXT,
                    error_msg TEXT,
                    processed_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
            """)  # 主键 file_hash：同一文件内容只对应一条历史

            # Create index on status for faster queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_status 
                ON ingestion_history(status)s
            """)  # 按 status 筛 success/failed 更快

            conn.commit()  # 提交 DDL 语句，创建表和索引
        finally:
            conn.close()  # 初始化完就关；业务方法里各自再 connect

    def compute_sha256(self, file_path: str) -> str: #判断当前这个文件有没有处理过
        """Compute SHA256 hash of file using chunked reading.
        
        Uses 64KB chunks to handle large files without loading entire
        file into memory.
        
        Args:
            file_path: Path to the file to hash.
            
        Returns:
            Hexadecimal SHA256 hash string (64 characters).
            
        Raises:
            FileNotFoundError: If file does not exist.
            IOError: If path is not a file or cannot be read.
        """
        path = Path(file_path)  # 统一成 Path 做存在性检查

        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        if not path.is_file():
            raise IOError(f"Path is not a file: {file_path}")  # 目录等不允许

        # Compute hash using chunked reading
        sha256_hash = hashlib.sha256()  # 空哈希对象

        try:
            with open(file_path, "rb") as f:  # 二进制读，避免编码
                # Read in 64KB chunks
                for chunk in iter(lambda: f.read(65536), b""):  # 每次 64KB，读完为止
                    sha256_hash.update(chunk)  # 流式更新，大文件也不爆内存
        except Exception as e:
            raise IOError(f"Failed to read file {file_path}: {e}")

        return sha256_hash.hexdigest()  # 64 位十六进制字符串

    def should_skip(self, file_hash: str) -> bool:
        """Check if file should be skipped.
        
        Only files with status='success' are skipped. Failed files
        can be retried.
        
        Args:
            file_hash: SHA256 hash of the file.
            
        Returns:
            True if file has status='success', False otherwise.
        """
        conn = sqlite3.connect(self.db_path)  # 短连接
        try:
            cursor = conn.execute(
                "SELECT status FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)  # 参数绑定防注入
            )
            result = cursor.fetchone()  # 一行或 None

            if result is None:
                return False  # 从没处理过：不跳过

            return result[0] == "success"  # 只有成功才算「可跳过」；failed 会再跑
        finally:
            conn.close()

    def mark_success(
        self, 
        file_hash: str, 
        file_path: str, 
        collection: Optional[str] = None
    ) -> None:
        """Mark file as successfully processed.
        
        Uses INSERT OR REPLACE for idempotent operation.
        
        Args:
            file_hash: SHA256 hash of the file.
            file_path: Original file path (for tracking).
            collection: Optional collection/namespace identifier.
            
        Raises:
            RuntimeError: If database operation fails.
        """
        now = datetime.now(timezone.utc).isoformat()  # 当前 UTC 时间 ISO 字符串

        conn = sqlite3.connect(self.db_path)
        try:
            # Check if record exists to preserve processed_at
            cursor = conn.execute(
                "SELECT processed_at FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()  # 是否已有记录

            if result:
                # Update existing record
                conn.execute("""
                    UPDATE ingestion_history 
                    SET file_path = ?,
                        status = 'success',
                        collection = ?,
                        error_msg = NULL,
                        updated_at = ?
                    WHERE file_hash = ?
                """, (file_path, collection, now, file_hash))  # 保留原 processed_at，只更新路径/集合/状态
            else:
                # Insert new record
                conn.execute("""
                    INSERT INTO ingestion_history 
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'success', ?, NULL, ?, ?)
                """, (file_hash, file_path, collection, now, now))  # 首次成功：processed 与 updated 都用 now

            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark success for {file_path}: {e}")
        finally:
            conn.close()

    def mark_failed(
        self, 
        file_hash: str, 
        file_path: str, 
        error_msg: str
    ) -> None:
        """Mark file processing as failed.
        
        Failed files are not skipped, allowing retries.
        
        Args:
            file_hash: SHA256 hash of the file.
            file_path: Original file path (for tracking).
            error_msg: Error message describing the failure.
            
        Raises:
            RuntimeError: If database operation fails.
        """
        now = datetime.now(timezone.utc).isoformat()

        conn = sqlite3.connect(self.db_path)
        try:
            # Check if record exists to preserve processed_at
            cursor = conn.execute(
                "SELECT processed_at FROM ingestion_history WHERE file_hash = ?",
                (file_hash,)
            )
            result = cursor.fetchone()

            if result:
                # Update existing record
                conn.execute("""
                    UPDATE ingestion_history 
                    SET file_path = ?,
                        status = 'failed',
                        error_msg = ?,
                        updated_at = ?
                    WHERE file_hash = ?
                """, (file_path, error_msg, now, file_hash))  # 已有行：记错因，不覆盖首次 processed_at
            else:
                # Insert new record
                conn.execute("""
                    INSERT INTO ingestion_history 
                    (file_hash, file_path, status, collection, error_msg, processed_at, updated_at)
                    VALUES (?, ?, 'failed', NULL, ?, ?, ?)
                """, (file_hash, file_path, error_msg, now, now))  # 新行：第一次就失败

            conn.commit()
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to mark failure for {file_path}: {e}")
        finally:
            conn.close()

    def remove_record(self, file_hash: str) -> bool:
        """Remove an ingestion record by its file hash.

        Args:
            file_hash: SHA256 hash identifying the record.

        Returns:
            True if a record was deleted, False if not found.
        """
        conn = sqlite3.connect(self.db_path)
        try:
            cursor = conn.execute(
                "DELETE FROM ingestion_history WHERE file_hash = ?",
                (file_hash,),  # 按主键删一条
            )
            conn.commit()
            return cursor.rowcount > 0  # 删了几行：1 表示删到了，0 表示本来就没有
        except sqlite3.Error as e:
            raise RuntimeError(f"Failed to remove record {file_hash}: {e}")
        finally:
            conn.close()

    def list_processed(
        self, collection: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """List successfully processed files.

        Args:
            collection: Optional collection filter.

        Returns:
            List of dicts with keys: file_hash, file_path, collection,
            processed_at, updated_at.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # 行可像 dict 一样按键取值
        try:
            query = (
                "SELECT file_hash, file_path, collection, processed_at, updated_at "
                "FROM ingestion_history WHERE status = 'success'"
            )  # 只列成功入库的文档
            params: list[str] = []
            if collection is not None:
                query += " AND collection = ?"  # 可选：只看某个集合
                params.append(collection)
            query += " ORDER BY processed_at ASC"  # 按首次成功时间排序

            cursor = conn.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]  # 转成普通 dict 列表给上层用
        finally:
            conn.close()
