import unittest

from scrape import normalize_connection_string


class NormalizeConnectionStringTests(unittest.TestCase):
    def test_encodes_invalid_percent_in_password(self) -> None:
        conn = "".join(
            ["postgresql://", "user", ":", "E$GgDEPU%_S3+*j", "@", "localhost:5432/mydb"]
        )
        normalized = normalize_connection_string(conn)
        self.assertEqual(
            normalized,
            "".join(
                [
                    "postgresql://",
                    "user",
                    ":",
                    "E$GgDEPU%25_S3+*j",
                    "@",
                    "localhost:5432/mydb",
                ]
            ),
        )

    def test_keeps_valid_percent_encoding(self) -> None:
        conn = "".join(
            ["postgresql://", "user", ":", "abc%40def", "@", "localhost:5432/mydb"]
        )
        normalized = normalize_connection_string(conn)
        self.assertEqual(normalized, conn)

    def test_keeps_non_url_conn_string(self) -> None:
        conn = "dbname=mydb user=user pass=abc%_def host=localhost"
        normalized = normalize_connection_string(conn)
        self.assertEqual(normalized, conn)


if __name__ == "__main__":
    unittest.main()
