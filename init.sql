-- One postgres instance, one database per demo.

CREATE DATABASE demo2;
CREATE DATABASE demo3;

\c demo2
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL
);

\c demo3
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE posts (
    id SERIAL PRIMARY KEY,
    content TEXT NOT NULL,
    category TEXT,
    embedding VECTOR(768) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
