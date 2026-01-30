# Deployment Guide

This guide covers deploying the Trading Orchestrator system in various environments.

## Table of Contents

- [Local Development](#local-development)
- [Docker Deployment](#docker-deployment)
- [LangGraph Cloud](#langgraph-cloud)
- [Kubernetes](#kubernetes)
- [Production Checklist](#production-checklist)
- [Monitoring](#monitoring)
- [Backup and Recovery](#backup-and-recovery)

## Local Development

### Prerequisites

- Python 3.11+
- Poetry or pip
- API keys (Anthropic or OpenAI)

### Setup

```bash
# Install dependencies
poetry install

# Configure environment
cp .env.example .env
# Edit .env with your API keys

# Run tests
poetry run pytest

# Run locally
poetry run python -m trading_orchestrator.cli
```

## Docker Deployment

### Quick Start

```bash
# Build image
docker build -t trading-orchestrator .

# Run container
docker run -d \
  --name trading-orchestrator \
  -e ANTHROPIC_API_KEY=your_key \
  -v $(pwd)/checkpoints:/app/checkpoints \
  -v $(pwd)/data:/app/data \
  trading-orchestrator
```

### Docker Compose

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f orchestrator

# Stop services
docker-compose down

# Rebuild after changes
docker-compose up -d --build
```

### Production Docker Configuration

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  orchestrator:
    build:
      context: .
      dockerfile: Dockerfile
    image: trading-orchestrator:latest
    container_name: trading-orchestrator-prod
    environment:
      - ANTHROPIC_API_KEY=${ANTHROPIC_API_KEY}
      - DATABASE_URL=postgresql://orchestrator:${POSTGRES_PASSWORD}@postgres:5432/trading_orchestrator
      - LOG_LEVEL=INFO
      - LOG_FORMAT=json
      - ENABLE_METRICS=true
    volumes:
      - /var/lib/trading-orchestrator/checkpoints:/app/checkpoints
      - /var/lib/trading-orchestrator/data:/app/data
      - /var/log/trading-orchestrator:/app/logs
    restart: always
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
    depends_on:
      postgres:
        condition: service_healthy
    networks:
      - trading-network

  postgres:
    image: postgres:15-alpine
    container_name: trading-orchestrator-postgres
    environment:
      - POSTGRES_DB=trading_orchestrator
      - POSTGRES_USER=orchestrator
      - POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
      - PGDATA=/var/lib/postgresql/data/pgdata
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "127.0.0.1:5432:5432"
    restart: always
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U orchestrator"]
      interval: 10s
      timeout: 5s
      retries: 5
    networks:
      - trading-network

volumes:
  postgres_data:
    driver: local

networks:
  trading-network:
    driver: bridge
```

Run with:

```bash
docker-compose -f docker-compose.prod.yml up -d
```

## LangGraph Cloud

### Prerequisites

```bash
pip install langgraph-cli
```

### Configuration

Create `langgraph.json`:

```json
{
  "dependencies": ["trading_orchestrator"],
  "graphs": {
    "trading_orchestrator": "./trading_orchestrator/graph.py:TradingOrchestrator"
  },
  "env": ".env",
  "python_version": "3.11"
}
```

### Deployment

```bash
# Login to LangGraph Cloud
langgraph login

# Deploy
langgraph deploy --env production

# View status
langgraph status

# View logs
langgraph logs --tail 100
```

### API Usage

After deployment, use the provided API endpoint:

```python
import httpx

async with httpx.AsyncClient() as client:
    response = await client.post(
        "https://your-deployment.langgraph.cloud/invoke",
        json={
            "task_description": "Develop EURUSD trading bot",
            "task_parameters": {
                "symbol": "EURUSD",
                "timeframe": "H1"
            }
        },
        headers={"Authorization": f"Bearer {api_key}"}
    )
    print(response.json())
```

## Kubernetes

### Deployment Manifest

Create `k8s/deployment.yaml`:

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: trading-orchestrator

---
apiVersion: v1
kind: ConfigMap
metadata:
  name: orchestrator-config
  namespace: trading-orchestrator
data:
  LOG_LEVEL: "INFO"
  LOG_FORMAT: "json"
  MAX_AGENT_ITERATIONS: "10"
  ENABLE_METRICS: "true"

---
apiVersion: v1
kind: Secret
metadata:
  name: orchestrator-secrets
  namespace: trading-orchestrator
type: Opaque
stringData:
  ANTHROPIC_API_KEY: "your_api_key_here"
  POSTGRES_PASSWORD: "your_postgres_password_here"

---
apiVersion: apps/v1
kind: Deployment
metadata:
  name: trading-orchestrator
  namespace: trading-orchestrator
spec:
  replicas: 2
  selector:
    matchLabels:
      app: trading-orchestrator
  template:
    metadata:
      labels:
        app: trading-orchestrator
    spec:
      containers:
      - name: orchestrator
        image: trading-orchestrator:latest
        imagePullPolicy: Always
        envFrom:
        - configMapRef:
            name: orchestrator-config
        - secretRef:
            name: orchestrator-secrets
        env:
        - name: DATABASE_URL
          value: "postgresql://orchestrator:$(POSTGRES_PASSWORD)@postgres-service:5432/trading_orchestrator"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        volumeMounts:
        - name: checkpoints
          mountPath: /app/checkpoints
        - name: data
          mountPath: /app/data
        livenessProbe:
          exec:
            command:
            - python
            - -c
            - "import trading_orchestrator; print('healthy')"
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          exec:
            command:
            - python
            - -c
            - "import trading_orchestrator; print('ready')"
          initialDelaySeconds: 10
          periodSeconds: 10
      volumes:
      - name: checkpoints
        persistentVolumeClaim:
          claimName: orchestrator-checkpoints-pvc
      - name: data
        persistentVolumeClaim:
          claimName: orchestrator-data-pvc

---
apiVersion: v1
kind: Service
metadata:
  name: orchestrator-service
  namespace: trading-orchestrator
spec:
  selector:
    app: trading-orchestrator
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: orchestrator-checkpoints-pvc
  namespace: trading-orchestrator
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 10Gi

---
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: orchestrator-data-pvc
  namespace: trading-orchestrator
spec:
  accessModes:
  - ReadWriteMany
  resources:
    requests:
      storage: 50Gi
```

### PostgreSQL StatefulSet

Create `k8s/postgres.yaml`:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: trading-orchestrator
spec:
  serviceName: postgres-service
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:15-alpine
        env:
        - name: POSTGRES_DB
          value: trading_orchestrator
        - name: POSTGRES_USER
          value: orchestrator
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: orchestrator-secrets
              key: POSTGRES_PASSWORD
        ports:
        - containerPort: 5432
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: [ "ReadWriteOnce" ]
      resources:
        requests:
          storage: 100Gi

---
apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: trading-orchestrator
spec:
  selector:
    app: postgres
  ports:
  - protocol: TCP
    port: 5432
    targetPort: 5432
  clusterIP: None
```

### Deploy to Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/postgres.yaml

# Check status
kubectl get pods -n trading-orchestrator

# View logs
kubectl logs -f deployment/trading-orchestrator -n trading-orchestrator

# Scale deployment
kubectl scale deployment/trading-orchestrator --replicas=3 -n trading-orchestrator
```

## Production Checklist

### Security

- [ ] API keys stored in secrets manager (not environment variables)
- [ ] Database credentials rotated regularly
- [ ] TLS/SSL enabled for all connections
- [ ] Network policies configured
- [ ] Container security scanning enabled
- [ ] Least privilege access controls

### Performance

- [ ] Resource limits configured
- [ ] Horizontal pod autoscaling enabled
- [ ] Database connection pooling configured
- [ ] Caching strategy implemented
- [ ] Request rate limiting in place

### Reliability

- [ ] Health checks configured
- [ ] Automated backups scheduled
- [ ] Disaster recovery plan documented
- [ ] Monitoring and alerting active
- [ ] Load testing completed
- [ ] Circuit breakers implemented

### Observability

- [ ] Structured logging enabled
- [ ] Metrics collection configured
- [ ] Distributed tracing active
- [ ] Dashboard created
- [ ] Alert thresholds defined
- [ ] On-call rotation established

## Monitoring

### Prometheus Configuration

Create `monitoring/prometheus.yml`:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'orchestrator'
    static_configs:
      - targets: ['orchestrator:9090']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']
```

### Grafana Dashboard

Import the provided dashboard from `monitoring/grafana/dashboards/orchestrator-dashboard.json`

Key metrics to monitor:

- Workflow completion rate
- Agent execution time
- Model training duration
- Validation pass rate
- API request latency
- Error rate by agent
- Resource utilization

## Backup and Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups"
DB_NAME="trading_orchestrator"

pg_dump -h postgres -U orchestrator $DB_NAME | gzip > $BACKUP_DIR/backup_$TIMESTAMP.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
```

Schedule with cron:

```cron
0 2 * * * /scripts/backup.sh
```

### Checkpoint Backup

```bash
# Rsync checkpoints to backup location
rsync -avz --delete /app/checkpoints/ /backups/checkpoints/
```

### Recovery Procedure

1. Stop orchestrator service
2. Restore database from backup
3. Restore checkpoint files
4. Verify data integrity
5. Restart orchestrator service
6. Monitor for issues

### Disaster Recovery

Full disaster recovery steps:

```bash
# 1. Provision new infrastructure
terraform apply

# 2. Restore database
gunzip -c backup_latest.sql.gz | psql -h new-postgres -U orchestrator trading_orchestrator

# 3. Restore checkpoints
rsync -avz /backups/checkpoints/ /app/checkpoints/

# 4. Deploy application
kubectl apply -f k8s/

# 5. Verify functionality
kubectl exec -it deployment/trading-orchestrator -- python -c "from trading_orchestrator import *"

# 6. Resume workflows
# Workflows with checkpoints will automatically resume
```

## Cost Optimization

### LLM API Costs

- Use caching for repeated queries
- Implement request batching where possible
- Use faster/cheaper models for simple tasks
- Monitor token usage per agent

### Infrastructure Costs

- Right-size container resources
- Use spot/preemptible instances for non-critical workloads
- Implement autoscaling to match demand
- Archive old checkpoints and logs

### Database Costs

- Implement data retention policies
- Use appropriate instance size
- Enable query optimization
- Consider read replicas for analytics

## Troubleshooting

### Common Issues

**Workflow hangs**
```bash
# Check logs
kubectl logs deployment/trading-orchestrator -n trading-orchestrator --tail=100

# Check resource usage
kubectl top pods -n trading-orchestrator

# Restart if needed
kubectl rollout restart deployment/trading-orchestrator -n trading-orchestrator
```

**Database connection issues**
```bash
# Test connection
kubectl exec -it deployment/trading-orchestrator -- python -c "from sqlalchemy import create_engine; engine = create_engine('$DATABASE_URL'); print(engine.execute('SELECT 1').fetchone())"

# Check postgres logs
kubectl logs statefulset/postgres -n trading-orchestrator
```

**Out of memory**
```bash
# Increase memory limits
kubectl set resources deployment/trading-orchestrator --limits=memory=8Gi -n trading-orchestrator

# Or scale horizontally
kubectl scale deployment/trading-orchestrator --replicas=4 -n trading-orchestrator
```

## Support

For deployment issues:
- Check logs first: `kubectl logs` or `docker-compose logs`
- Review resource usage: `kubectl top` or `docker stats`
- Consult troubleshooting section above
- Open GitHub issue with logs and configuration
