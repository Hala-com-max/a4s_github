#!/bin/bash

# Entrypoint script for A4S Evaluation Service
# Supports multiple modes: server, worker, or combined

set -e

# Function to wait for dependencies
wait_for_dependencies() {
    # Wait for Redis
    if [ -n "$REDIS_HOST" ]; then
        echo "Waiting for Redis at $REDIS_HOST:${REDIS_PORT:-6379}..."
        timeout 30 bash -c "until nc -z $REDIS_HOST ${REDIS_PORT:-6379}; do sleep 1; done" || {
            echo "Redis not reachable at $REDIS_HOST:${REDIS_PORT:-6379}"
        }
    fi
    
    # Wait for RabbitMQ
    local mq_host=$(echo "${MQ_AMQP_ENDPOINT:-localhost}" | sed -E 's/^[a-z]+:\/\/([^:\/]+).*/\1/')
    if [ -n "$mq_host" ]; then
        echo "Waiting for RabbitMQ at $mq_host..."
        timeout 30 bash -c "until nc -z $mq_host 5671; do sleep 1; done" || {
            echo "RabbitMQ not reachable at $mq_host:5671"
        }
    fi
}

# Function to start FastAPI server
start_server() {
    echo "Starting FastAPI evaluation server..."
    exec uvicorn a4s_eval.main:app --host 0.0.0.0 --port 8001
}

# Function to start Celery worker
start_worker() {
    echo "Starting Celery worker..."
    exec celery -A a4s_eval.celery_worker worker --loglevel=info --concurrency=1 --hostname=worker@%h
}

# Function to start both server and worker
start_combined() {
    echo "Starting combined server and worker..."
    
    # Start Celery worker in background
    echo "Starting Celery worker in background..."
    celery -A a4s_eval.celery_worker worker --loglevel=info --concurrency=1 --hostname=worker@%h &
    
    # Wait a moment for worker to start
    sleep 5
    
    # Start FastAPI server in foreground
    echo "Starting FastAPI server..."
    exec uvicorn a4s_eval.main:app --host 0.0.0.0 --port 8001
}

# Main function
main() {
    # Wait for dependencies
    wait_for_dependencies
    
    # Start based on mode
    case "${1:-server}" in
        "server")
            start_server
            ;;
        "worker")
            start_worker
            ;;
        "combined")
            start_combined
            ;;
    esac
}

# Execute main function with all arguments
main "$@"
