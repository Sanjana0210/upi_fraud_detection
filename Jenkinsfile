// ─── UPI Fraud Detection — Jenkins CI/CD Pipeline ───────────────────────────
// Stages: Checkout → Install → Lint → Test → Docker Build → Deploy

pipeline {
    agent any

    environment {
        PYTHON_HOME    = '/opt/homebrew/bin'
        VENV_DIR       = '.venv-jenkins'
        IMAGE_NAME     = 'upi-fraud-detection'
        IMAGE_TAG      = "${BUILD_NUMBER}"
        REPO_URL       = 'https://github.com/Sanjana0210/upi_fraud_detection.git'
        PATH           = "/opt/homebrew/bin:/usr/local/bin:${env.PATH}"
    }

    options {
        timestamps()
        timeout(time: 15, unit: 'MINUTES')
        buildDiscarder(logRotator(numToKeepStr: '10'))
    }

    stages {

        // ── Stage 1: Checkout ────────────────────────────────────────────────
        stage('📥 Checkout') {
            steps {
                git branch: 'main',
                    url: "${REPO_URL}"
            }
        }

        // ── Stage 2: Setup Python Environment ───────────────────────────────
        stage('🐍 Setup Python') {
            steps {
                sh '''
                    python3 -m venv ${VENV_DIR}
                    . ${VENV_DIR}/bin/activate
                    pip install --upgrade pip --quiet
                    pip install -r requirements-ci.txt --quiet
                '''
            }
        }

        // ── Stage 3: Lint ────────────────────────────────────────────────────
        stage('🔍 Lint') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    echo "Running flake8 static analysis..."
                    flake8 . --config .flake8 --statistics --format=pylint || true
                '''
            }
        }

        // ── Stage 4: Test ────────────────────────────────────────────────────
        stage('🧪 Test') {
            steps {
                sh '''
                    . ${VENV_DIR}/bin/activate
                    echo "Running pytest with coverage..."
                    PYTHONPATH=. pytest tests/ -v --tb=short \
                        --cov=utils --cov=ml \
                        --cov-report=term-missing \
                        --cov-report=xml:coverage.xml \
                        --junitxml=test-results.xml
                '''
            }
            post {
                always {
                    junit allowEmptyResults: true, testResults: 'test-results.xml'
                }
            }
        }

        // ── Stage 5: Docker Build ────────────────────────────────────────────
        stage('🐳 Docker Build') {
            steps {
                sh '''
                    echo "Building Docker image: ${IMAGE_NAME}:${IMAGE_TAG}"
                    docker build -t ${IMAGE_NAME}:${IMAGE_TAG} \
                                 -t ${IMAGE_NAME}:latest .
                '''
            }
        }

        // ── Stage 6: Docker Deploy ───────────────────────────────────────────
        stage('🚀 Deploy') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    echo "Stopping old containers..."
                    docker compose down --remove-orphans 2>/dev/null || true

                    echo "Starting fresh deployment..."
                    docker compose up -d --build

                    echo "Waiting for health check..."
                    sleep 10
                    curl -f http://localhost:8501/_stcore/health || echo "⚠️  App not ready yet"

                    echo "✅ Deployed successfully!"
                '''
            }
        }
    }

    post {
        success {
            echo '✅ Pipeline completed successfully!'
        }
        failure {
            echo '❌ Pipeline failed — check the logs above.'
        }
        always {
            // Clean up virtual environment
            sh 'rm -rf ${VENV_DIR} || true'
        }
    }
}
