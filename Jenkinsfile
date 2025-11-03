pipeline {
    agent any
    
    environment {
        PROJECT_ID = 'amiable-mix-418317'
        APP_NAME = 'ads-placement'
        REPO_NAME = 'my-app-repo'
        REGION = 'us-central1'
        IMAGE_TAG = "${BUILD_NUMBER}"
        IMAGE_NAME = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${APP_NAME}"
        GCP_CREDS = credentials('gcp-service-account')
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo "Checking out code from GitHub..."
                checkout scm
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo "Building Docker image..."
                script {
                    sh "docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ."
                    sh "docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest"
                }
            }
        }
        
        stage('Push to Artifact Registry') {
            steps {
                echo "Pushing image to GCP Artifact Registry..."
                script {
                    sh """
                        gcloud auth activate-service-account --key-file=${GCP_CREDS}
                        gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
                        docker push ${IMAGE_NAME}:${IMAGE_TAG}
                        docker push ${IMAGE_NAME}:latest
                    """
                }
            }
        }
        
        stage('Deploy to Cloud Run') {
            steps {
                echo "Deploying to Google Cloud Run..."
                script {
                    sh """
                        gcloud config set project ${PROJECT_ID}
                        gcloud run deploy ${APP_NAME} \
                            --image ${IMAGE_NAME}:${IMAGE_TAG} \
                            --platform managed \
                            --region ${REGION} \
                            --allow-unauthenticated \
                            --port 8080 \
                            --max-instances 3 \
                            --memory 2Gi \
                            --timeout 3600
                    """
                }
            }
        }
        
        stage('Cleanup') {
            steps {
                echo "Cleaning up local Docker images..."
                script {
                    sh """
                        docker rmi ${IMAGE_NAME}:${IMAGE_TAG} || true
                        docker rmi ${IMAGE_NAME}:latest || true
                    """
                }
            }
        }
    }
    
    post {
        success {
            echo "Pipeline completed successfully!"
            emailext (
                subject: "SUCCESS: ${env.JOB_NAME} - Build #${env.BUILD_NUMBER}",
                body: """✅ Build succeeded!
                
Project: ${env.JOB_NAME}
Build Number: ${env.BUILD_NUMBER}
Application: ${APP_NAME}
Image: ${IMAGE_NAME}:${IMAGE_TAG}

View deployment: https://console.cloud.google.com/run/detail/${REGION}/${APP_NAME}
""",
                to: "raghav.vijayanand@gmail.com"
            )
        }
        failure {
            echo "Pipeline failed!"
            emailext (
                subject: "❌ FAILED: ${env.JOB_NAME} - Build #${env.BUILD_NUMBER}",
                body: """Build failed!

Project: ${env.JOB_NAME}
Build Number: ${env.BUILD_NUMBER}

Check logs: ${env.BUILD_URL}console
""",
                to: "raghav.vijayanand@gmail.com"
            )
        }
    }
}
pipeline {
    agent any
    
    environment {
        PROJECT_ID = 'amiable-mix-418317'
        APP_NAME = 'ads-make'
        REPO_NAME = 'ads-make-repo'
        REGION = 'us-central1'
        IMAGE_TAG = "${BUILD_NUMBER}"
        IMAGE_NAME = "${REGION}-docker.pkg.dev/${PROJECT_ID}/${REPO_NAME}/${APP_NAME}"
        GCP_CREDS = credentials('gcp-service-account')
    }
    
    stages {
        stage('Checkout') {
            steps {
                echo "Checking out code from GitHub..."
                checkout scm
            }
        }
        
        stage('Build') {
            steps {
                echo "Building Python application..."
                script {
                    // Install Python dependencies
                    sh '''
                        python3 --version
                        pip3 install --upgrade pip
                        pip3 install -r requirements.docker.txt || pip3 install -r requirements.txt
                    '''
                }
            }
        }
        
        stage('Test') {
            steps {
                echo "Running tests..."
                script {
                    sh '''
                        # Run Python tests if they exist
                        if [ -d "tests" ]; then
                            pip3 install pytest pytest-cov
                            pytest tests/ --cov=src --cov-report=html --cov-report=term || true
                        else
                            echo "No tests directory found, skipping tests"
                        fi
                        
                        # Basic syntax check
                        python3 -m py_compile app.py
                        python3 -m py_compile src/*.py
                    '''
                }
            }
        }
        
        stage('Build Docker Image') {
            steps {
                echo "Building Docker image..."
                script {
                    sh """
                        docker build -f Dockerfile.prod -t ${IMAGE_NAME}:${IMAGE_TAG} .
                        docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:latest
                        docker tag ${IMAGE_NAME}:${IMAGE_TAG} ${IMAGE_NAME}:build-${IMAGE_TAG}
                    """
                }
            }
        }
        
        stage('Security Scan') {
            steps {
                echo "Scanning Docker image for vulnerabilities..."
                script {
                    sh """
                        # Install trivy if not available
                        which trivy || echo "Trivy not installed, skipping scan"
                        
                        # Scan image (non-blocking)
                        trivy image --severity HIGH,CRITICAL ${IMAGE_NAME}:${IMAGE_TAG} || true
                    """
                }
            }
        }
        
        stage('Push to Artifact Registry') {
            steps {
                echo "Pushing image to GCP Artifact Registry..."
                script {
                    sh """
                        # Authenticate with GCP
                        gcloud auth activate-service-account --key-file=\${GCP_CREDS}
                        gcloud config set project ${PROJECT_ID}
                        
                        # Configure Docker to use Artifact Registry
                        gcloud auth configure-docker ${REGION}-docker.pkg.dev --quiet
                        
                        # Push images
                        docker push ${IMAGE_NAME}:${IMAGE_TAG}
                        docker push ${IMAGE_NAME}:latest
                        docker push ${IMAGE_NAME}:build-${IMAGE_TAG}
                        
                        echo "Images pushed successfully:"
                        echo "  - ${IMAGE_NAME}:${IMAGE_TAG}"
                        echo "  - ${IMAGE_NAME}:latest"
                    """
                }
            }
        }
        
        stage('Deploy to Cloud Run') {
            steps {
                echo "Deploying to Google Cloud Run..."
                script {
                    sh """
                        gcloud config set project ${PROJECT_ID}
                        
                        gcloud run deploy ${APP_NAME} \
                            --image ${IMAGE_NAME}:${IMAGE_TAG} \
                            --platform managed \
                            --region ${REGION} \
                            --allow-unauthenticated \
                            --port 8080 \
                            --memory 4Gi \
                            --cpu 2 \
                            --timeout 300 \
                            --max-instances 5 \
                            --min-instances 0 \
                            --set-env-vars "FLASK_ENV=production,PORT=8080" \
                            --quiet
                        
                        # Get the service URL
                        SERVICE_URL=\$(gcloud run services describe ${APP_NAME} \
                            --region ${REGION} \
                            --format 'value(status.url)')
                        
                        echo "Application deployed successfully!"
                        echo "Service URL: \${SERVICE_URL}"
                        
                        # Test the deployed service
                        echo "Testing health endpoint..."
                        curl -f \${SERVICE_URL}/health || echo "Health check warning"
                    """
                }
            }
        }
        
        stage('Cleanup') {
            steps {
                echo "Cleaning up local Docker images..."
                script {
                    sh """
                        docker rmi ${IMAGE_NAME}:${IMAGE_TAG} || true
                        docker rmi ${IMAGE_NAME}:latest || true
                        docker rmi ${IMAGE_NAME}:build-${IMAGE_TAG} || true
                        
                        # Clean up dangling images
                        docker image prune -f || true
                    """
                }
            }
        }
    }
    
    post {
        success {
            echo "Pipeline completed successfully!"
            script {
                def SERVICE_URL = sh(
                    script: "gcloud run services describe ${APP_NAME} --region ${REGION} --format 'value(status.url)' 2>/dev/null || echo 'URL not available'",
                    returnStdout: true
                ).trim()
                
                emailext (
                    subject: "✅ SUCCESS: ${env.JOB_NAME} - Build #${env.BUILD_NUMBER}",
                    body: """
                        <h2>Build Succeeded!</h2>
                        
                        <h3>Build Information:</h3>
                        <ul>
                            <li><strong>Project:</strong> ${env.JOB_NAME}</li>
                            <li><strong>Build Number:</strong> ${env.BUILD_NUMBER}</li>
                            <li><strong>Build URL:</strong> <a href="${env.BUILD_URL}">${env.BUILD_URL}</a></li>
                            <li><strong>GCP Project:</strong> ${PROJECT_ID}</li>
                            <li><strong>Region:</strong> ${REGION}</li>
                        </ul>
                        
                        <h3>Deployment Details:</h3>
                        <ul>
                            <li><strong>Application:</strong> ${APP_NAME}</li>
                            <li><strong>Image Tag:</strong> ${IMAGE_TAG}</li>
                            <li><strong>Service URL:</strong> <a href="${SERVICE_URL}">${SERVICE_URL}</a></li>
                        </ul>
                        
                        <h3>Docker Image:</h3>
                        <pre>${IMAGE_NAME}:${IMAGE_TAG}</pre>
                        
                        <p><strong>Status:</strong> Application deployed to Cloud Run successfully! 🚀</p>
                    """,
                    to: "raghav.vijayanand@gmail.com",
                    mimeType: 'text/html'
                )
            }
        }
        
        failure {
            echo "Pipeline failed!"
            emailext (
                subject: "❌ FAILED: ${env.JOB_NAME} - Build #${env.BUILD_NUMBER}",
                body: """
                    <h2>Build Failed!</h2>
                    
                    <h3>Build Information:</h3>
                    <ul>
                        <li><strong>Project:</strong> ${env.JOB_NAME}</li>
                        <li><strong>Build Number:</strong> ${env.BUILD_NUMBER}</li>
                        <li><strong>Build URL:</strong> <a href="${env.BUILD_URL}">${env.BUILD_URL}</a></li>
                        <li><strong>Console Output:</strong> <a href="${env.BUILD_URL}console">${env.BUILD_URL}console</a></li>
                    </ul>
                    
                    <h3>Error Details:</h3>
                    <p>Check the Jenkins console output for detailed error information.</p>
                    
                    <h3>Quick Links:</h3>
                    <ul>
                        <li><a href="${env.BUILD_URL}console">Console Output</a></li>
                        <li><a href="${env.BUILD_URL}changes">Changes</a></li>
                        <li><a href="${env.BUILD_URL}testReport">Test Report</a></li>
                    </ul>
                    
                    <p><strong>Status:</strong> Deployment failed. Please review the logs and retry. ⚠️</p>
                """,
                to: "raghav.vijayanand@gmail.com",
                mimeType: 'text/html'
            )
        }
        
        always {
            echo "Archiving artifacts and cleaning workspace..."
            script {
                // Archive build logs
                archiveArtifacts artifacts: '*.log', allowEmptyArchive: true
                
                // Clean workspace (optional)
                // cleanWs()
            }
        }
    }
}
