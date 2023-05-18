from typing import Tuple
from opensearchpy import OpenSearch


class Searcher:

    def __init__(self,
                 host: str = "localhost",
                 port: int = 9200,
                 username: str = "admin",
                 password: str = "admin",):

        self.host = host
        self.port = port
        self.username = username
        self.password = password

        self.client = OpenSearch(
                hosts=[{"host": host, "port": port}],
                http_auth=(username, password),
                use_ssl=True,
                verify_certs=False,
                ssl_assert_hostname=False,
                ssl_show_warn=False,
        )


    def is_alive(self):
        return self.client.ping()

    def search(self,
               query: str,
               size: int = 3,
               index: str = "minecraft",
               fields: Tuple[str] = ("title^2", "texts")):

        query_dict = {
            "size": size,
            "query": {
                "query_string": {
                    "query": query,
                    "fields": fields,
                }
            },
        }

        response = self.client.search(body=query_dict, index=index)

        output = [
            {
                "texts": "\n".join(response["hits"]["hits"][i]["_source"]["texts"]),
            }
            for i in range(len(response["hits"]["hits"]))
        ]
        return output


if __name__ == "__main__":
    searcher = Searcher()
    print(searcher.is_alive())
    search_results = searcher.search("Make a diamond pickaxe?")
    print(len(search_results))
    print(search_results[0]["texts"])
    print(search_results[1]["texts"])