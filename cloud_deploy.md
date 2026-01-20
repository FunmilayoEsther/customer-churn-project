## **cloud_deploy.md (ready-to-copy)**

```markdown
# Deployment Instructions

This API can be run **locally** or on **Render**.

---

## Local Docker Deployment

1. Build Docker image:
```bash
docker build -t churn-predictor .

2. Run container:
``` bash
docker run -p 9696:9696 churn-predictor

3. Test API
```bash
curl http://localhost:9696/health
curl -X POST http://localhost:9696/predict \
-H "Content-Type: application/json" \
-d '{ ... }'  # single JSON object

4) Test deployed API
``` bash
curl https://your-app.onrender.com/health
curl -X POST https://your-app.onrender.com/predict \
-H "Content-Type: application/json" \
-d '{ ... }'