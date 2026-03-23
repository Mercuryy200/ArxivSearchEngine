-- Add expression index so duplicate-check queries on metadata->>'url' don't
-- cause full table scans (which time out as the documents table grows).
CREATE INDEX IF NOT EXISTS idx_documents_metadata_url
    ON documents ((metadata->>'url'));
