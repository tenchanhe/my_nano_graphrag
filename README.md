# Project Name

This project is built upon **nano graphrag**, extending its functionality to enhance performance and usability. Before using this repository, you must first install **nano graphrag**.

## Installation

```bash
pip install -r requirements.txt
```


| Index Step       | graphRAG(microsoft)                      | nano graphrag                               | lightRAG | try            |
| ---------------- | ---------------------------------------- | ------------------------------------------- | -------- | -------------- |
| chunk docs       |                                          | overlap_token_size=128, max_token_size=1024 |          | semantic chunk |
| extract entity   | extract entities prompt                  | extract entities prompt                     |          |                |
|                  | continue extract entities prompt         | continue extract entities prompt            |          |                |
|                  | no entity eliment                        | no entity eliment                           |          |                |
| entity type      | "organization", "person", "geo", "event" | "organization", "person", "geo", "event"    |          |                |
| community report | clustering (hierarchical_leiden)         | clustering (hierarchical_leiden)            |          |                |
|                  |                                          |                                             |          |                |
 


| Local Query Step        | graphRAG(microsoft) | nano graphrag       | lightRAG | try                                |
| ----------------------- | ------------------- | ------------------- | -------- | ---------------------------------- |
| entity similarity score |                     | cosine similarity   |          | LLM？ dense embedding alternative? |
| response information    |                     |                     |          |                                    |
|                         |                     |                     |          |                                    |
| response prompt         |                     | Multiple Paragraphs |          | Just Simple Answer                 |
|                         |                     |                     |          |                                    |




| Global Query Step          | graphRAG(microsoft) | nano graphrag       | lightRAG | try |
| -------------------------- | ------------------- | ------------------- | -------- | --- |
| community similarity score |                     | LLM rating          |          |     |
| response prompt            |                     | Multiple Paragraphs |          |     |
|                            |                     |                     |          |     |
