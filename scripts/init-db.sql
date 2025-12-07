-- =============================================================================
-- Database Initialization Script
-- =============================================================================
-- This script runs when the PostgreSQL container is first created

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Grant privileges
GRANT ALL PRIVILEGES ON DATABASE workforce_simulator TO postgres;

-- Create application schema (optional - tables are created by SQLAlchemy)
-- The actual tables will be created by the application on startup

-- Log initialization
DO $$
BEGIN
    RAISE NOTICE 'Database initialization complete';
END $$;
