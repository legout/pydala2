"""
Security tests for PyDala2
"""
import pytest
import tempfile
import os
from pydala.helpers.security import (
    escape_sql_identifier,
    escape_sql_literal,
    validate_partition_name,
    validate_partition_value,
    sanitize_filter_expression,
    validate_path,
    safe_join,
    sanitize_filename,
)


class TestSqlSecurity:
    """Test SQL security utilities."""
    
    def test_escape_sql_identifier(self):
        """Test SQL identifier escaping."""
        # Valid identifiers
        assert escape_sql_identifier("column_name") == '"column_name"'
        assert escape_sql_identifier("_column") == '"_column"'
        assert escape_sql_identifier("column123") == '"column123"'
        
        # Invalid identifiers should raise
        with pytest.raises(ValueError):
            escape_sql_identifier("")
        with pytest.raises(ValueError):
            escape_sql_identifier("123column")  # Starts with number
        with pytest.raises(ValueError):
            escape_sql_identifier("column-name")  # Invalid char
    
    def test_escape_sql_literal(self):
        """Test SQL literal escaping."""
        assert escape_sql_literal(None) == "NULL"
        assert escape_sql_literal("test") == "'test'"
        assert escape_sql_literal("test's") == "'test''s'"  # Escaped quote
        assert escape_sql_literal(123) == "123"
        assert escape_sql_literal(True) == "TRUE"
        assert escape_sql_literal(False) == "FALSE"
    
    def test_validate_partition_name(self):
        """Test partition name validation."""
        # Valid names
        assert validate_partition_name("year") is True
        assert validate_partition_name("month") is True
        assert validate_partition_name("day_of_month") is True
        
        # Invalid names
        assert validate_partition_name("") is False
        assert validate_partition_name("year/month") is False
        assert validate_partition_name("year;DROP TABLE") is False
        assert validate_partition_name("a" * 256) is False
    
    def test_validate_partition_value(self):
        """Test partition value validation."""
        # Valid values
        assert validate_partition_value("2023") is True
        assert validate_partition_value(2023) is True
        assert validate_partition_value(True) is True
        assert validate_partition_value(None) is True
        
        # Invalid values
        assert validate_partition_value("../../../etc/passwd") is False
        assert validate_partition_value("value\0withnull") is False
        assert validate_partition_value("value\nwithnewline") is False
        assert validate_partition_value("a" * 1025) is False
    
    def test_sanitize_filter_expression(self):
        """Test filter expression sanitization."""
        # Valid expressions
        assert sanitize_filter_expression("column > 100") == "column > 100"
        assert sanitize_filter_expression("column = 'test'") == "column = 'test'"
        
        # Dangerous content should be removed
        assert "-- comment" not in sanitize_filter_expression("column > 100 -- comment")
        assert "/*" not in sanitize_filter_expression("column > 100 /* comment */")
        
        # Unbalanced quotes should raise
        with pytest.raises(ValueError):
            sanitize_filter_expression("column = 'test")


class TestPathSecurity:
    """Test path security utilities."""
    
    def test_validate_path(self):
        """Test path validation."""
        # Valid paths
        assert validate_path("valid/path") == "valid/path"
        assert validate_path("file.txt") == "file.txt"
        
        # Invalid paths
        with pytest.raises(ValueError):
            validate_path("")
        with pytest.raises(ValueError):
            validate_path("../etc/passwd")
        with pytest.raises(ValueError):
            validate_path("..\windows\system32")
        with pytest.raises(ValueError):
            validate_path("/absolute/path")
        
        # With base path
        base = "/safe/base"
        assert validate_path("subdir", base) == "subdir"
        with pytest.raises(ValueError):
            validate_path("../../../etc/passwd", base)
    
    def test_safe_join(self):
        """Test safe path joining."""
        base = "/safe/base"
        
        # Valid joins
        assert safe_join(base, "subdir") == "/safe/base/subdir"
        assert safe_join(base, "file.txt") == "/safe/base/file.txt"
        
        # Invalid joins should raise
        with pytest.raises(ValueError):
            safe_join(base, "../etc/passwd")
        with pytest.raises(ValueError):
            safe_join(base, "", "../malicious")
    
    def test_sanitize_filename(self):
        """Test filename sanitization."""
        # Valid filenames
        assert sanitize_filename("document.txt") == "document.txt"
        assert sanitize_filename("data_file.csv") == "data_file.csv"
        
        # Invalid characters removed
        assert sanitize_filename("doc<>ument.txt") == "document.txt"
        assert sanitize_filename("file/name.txt") == "filename.txt"
        
        # Edge cases
        assert sanitize_filename("") == "unnamed"
        assert sanitize_filename("   .   ") == "unnamed"
        assert len(sanitize_filename("a" * 300)) <= 255


class TestCredentialHandling:
    """Test credential handling security."""
    
    def test_get_credentials_redaction(self):
        """Test credential redaction."""
        from pydala.filesystem import get_credentials_from_fsspec
        from unittest.mock import Mock
        
        # Mock filesystem with credentials
        mock_fs = Mock()
        mock_fs.protocol = ["s3"]
        mock_creds = Mock()
        mock_creds.access_key = "AKIAIOSFODNN7EXAMPLE"
        mock_creds.secret_key = "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"
        mock_creds.token = "session_token_value"
        mock_fs.s3 = Mock()
        mock_fs.s3._get_credentials.return_value = mock_creds
        mock_fs.s3._endpoint = Mock()
        mock_fs.s3._endpoint.host = "s3.amazonaws.com"
        
        # Test redaction (default)
        creds = get_credentials_from_fsspec(mock_fs, redact_secrets=True)
        assert "REDACTED" in creds["access_key"]
        assert "REDACTED" in creds["secret_key"]
        assert creds["endpoint_override"] == "s3.amazonaws.com"
        
        # Test non-redacted
        creds = get_credentials_from_fsspec(mock_fs, redact_secrets=False)
        assert creds["access_key"] == "AKIAIOSFODNN7EXAMPLE"
        assert creds["secret_key"] == "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])