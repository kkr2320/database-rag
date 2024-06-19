#DFS Financial AI Analyzer App

create table dfs_financial_documents ( id uuid PRIMARY KEY DEFAULT uuid_generate_v1() , doc_type varchar(20) , doc_date date , doc_page_content text , embeddings vector(1024) , additional_metadata jsonb) ;

CREATE INDEX ON dfs_financial_documents USING hnsw (embeddings vector_cosine_ops);

CREATE INDEX ON dfs_financial_documents ( doc_type ) ;

CREATE INDEX ON dfs_financial_documents ( doc_date ) ;

CREATE INDEX ON dfs_financial_documents ( doc_type , doc_date ) ;
