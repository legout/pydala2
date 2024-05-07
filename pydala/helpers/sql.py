import sqlparse
from sqlglot import exp, parse_one

# def get_table_names(sql_query):
#     table_names = set()
#     parsed_query = sqlparse.parse(sql_query)[0]

#     for token in parsed_query.tokens:
#         if isinstance(token, sqlparse.sql.IdentifierList):
#             for identifier in token.get_identifiers():
#                 table_names.add(identifier.get_real_name())
#         elif isinstance(token, sqlparse.sql.Identifier):
#             table_names.add(token.get_real_name())

#     return table_names


def get_table_names(sql_query):
    return [table.name for table in parse_one(sql_query).find_all(exp.Table)]


def replace_table_names_with_file_paths(sql_query, table_to_path_mapping):
    parsed_query = sqlparse.parse(sql_query)[0]
    new_tokens = []

    for token in parsed_query.tokens:
        if isinstance(token, sqlparse.sql.Identifier):
            table_name = token.get_real_name()
            if table_name in table_to_path_mapping:
                file_path, file_type, hive = table_to_path_mapping[table_name]
                read_function = "read_csv"  # default to CSV
                if file_type == "parquet":
                    read_function = "read_parquet"
                elif file_type == "json":
                    read_function = "read_json"
                # add more file types if needed
                new_token = sqlparse.sql.Token(
                    "Identifier",
                    f"{read_function}('{file_path.rstrip('/')}/**/*.{file_type}', "
                    f"hive_partitioning={hive}) AS {table_name}",
                )
                new_tokens.append(new_token)
            else:
                new_tokens.append(token)
        else:
            new_tokens.append(token)

    parsed_query.tokens = new_tokens
    return str(parsed_query)
