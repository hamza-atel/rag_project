# Operations

Backups
- Daily pg_dump: `pg_dump -U rag_user rag > /backups/rag-$(date +%F).sql`

Service
- Run Ollama as systemd service (example unit file in docs/)

Monitoring
- Disk usage alert for DB and models
- Simple cron to check free disk and restart services

