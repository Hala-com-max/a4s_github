# from typing import Any

from fastapi import APIRouter
from a4s_eval.celery_tasks import poll_and_run_evaluation
from a4s_eval.utils.logging import get_logger
# from a4s_eval.utils import env

router = APIRouter()
logger = get_logger()


@router.get("/evaluate")
async def evaluate() -> dict[str, str]:
    """Trigger evaluation of pending evaluatsions"""
    try:
        logger.debug("=== EVALUATE ENDPOINT START ===")
        logger.debug("1. About to call poll_and_run_evaluation.delay()")

        # Launch the evaluation task asynchronously
        task = poll_and_run_evaluation.delay()

        logger.debug("2. poll_and_run_evaluation.delay() completed successfully")
        logger.debug(f"3. Task ID: {task.id}")
        logger.debug("4. About to return response")

        return {
            "message": "Evaluation started.",
            "task_id": str(task.id),
            "status": "queued",
        }
    except Exception as e:
        logger.error(f"ERROR in evaluate endpoint: {str(e)}")
        logger.error(f"Exception type: {type(e)}")
        import traceback

        logger.error(f"Traceback: {traceback.format_exc()}")
        return {"message": f"Failed to start evaluation: {str(e)}", "status": "error"}
    finally:
        logger.debug("=== EVALUATE ENDPOINT END ===")


# @router.get("/evaluate/test-api-connection")
# async def test_api_connection() -> dict[str, str]:
#     """Test connection to main API"""
#     import requests

#     try:
#         resp = requests.get(f"{env.API_URL}/health", timeout=10)
#         return {
#             "status": "success",
#             "response_status": str(resp.status_code),
#             "message": "API connection successful"
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"API connection failed: {str(e)}"
#         }

# @router.get("/evaluate/test-celery")
# async def test_celery() -> dict[str, Any]:
#     """Test Celery broker/backend connectivity"""
#     from a4s_eval.celery_app import celery_app

#     try:
#         # Test broker connection
#         broker_connected = celery_app.connection().connected
#         workers = celery_app.control.inspect().active()

#         return {
#             "status": "success",
#             "celery_connected": str(broker_connected),
#             "workers": str(workers) if workers else "No workers found",
#             "message": "Celery connectivity test completed"
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"Celery connectivity test failed: {str(e)}"
#         }

# @router.get("/evaluate/test-rabbitmq")
# async def test_rabbitmq() -> dict[str, str]:
#     """Test RabbitMQ connectivity via Celery"""
#     from a4s_eval.celery_app import celery_app

#     try:
#         # Test RabbitMQ connection via Celery
#         with celery_app.connection() as conn:
#             connected = conn.connected
#             return {
#                 "status": "success",
#                 "rabbitmq_connected": str(connected),
#                 "message": "RabbitMQ connection test completed"
#             }
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"RabbitMQ connection test failed: {str(e)}"
#         }

# @router.get("/evaluate/test-simple-task")
# async def test_simple_task_endpoint() -> dict[str, str]:
#     """Test simple Celery task launch"""
#     import logging
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     try:
#         logger.info("=== TEST SIMPLE TASK ENDPOINT START ===")
#         logger.info("1. About to import test_simple_task")

#         logger.info("2. About to call test_simple_task.delay()")
#         task = test_simple_task.delay()

#         logger.info("3. test_simple_task.delay() completed successfully")
#         logger.info(f"4. Task ID: {task.id}")

#         return {
#             "status": "success",
#             "message": "Simple task launched successfully",
#             "task_id": str(task.id)
#         }
#     except Exception as e:
#         logger.error(f"ERROR in test_simple_task_endpoint: {str(e)}")
#         logger.error(f"Exception type: {type(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         return {
#             "status": "error",
#             "message": f"Simple task launch failed: {str(e)}"
#         }
#     finally:
#         logger.info("=== TEST SIMPLE TASK ENDPOINT END ===")

# @router.get("/evaluate/test-registered-tasks")
# async def test_registered_tasks() -> dict[str, str]:
#     """Test endpoint to check which tasks are registered with Celery."""
#     try:
#         from a4s_eval.celery_app import celery_app

#         # Get all registered tasks
#         registered_tasks = list(celery_app.tasks.keys())

#         # Check if our specific tasks are registered
#         poll_task_registered = "a4s_eval.celery_tasks.poll_and_run_evaluation" in registered_tasks
#         finalize_task_registered = "a4s_eval.celery_tasks.finalize_evaluation" in registered_tasks
#         test_task_registered = "a4s_eval.celery_tasks.test_simple_task" in registered_tasks

#         return {
#             "status": "success",
#             "message": "Task registration check completed",
#             "total_tasks": str(len(registered_tasks)),
#             "poll_task_registered": str(poll_task_registered),
#             "finalize_task_registered": str(finalize_task_registered),
#             "test_task_registered": str(test_task_registered),
#             "all_tasks": str(registered_tasks)
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"Failed to check registered tasks: {str(e)}"
#         }

# @router.get("/evaluate/test-api-url")
# async def test_api_url() -> dict[str, str]:
#     """Test endpoint to check API URL configuration."""
#     try:
#         from a4s_eval.utils.env import API_URL, API_PREFIX, API_URL_PREFIX

#         return {
#             "status": "success",
#             "message": "API URL configuration check completed",
#             "api_url": str(API_URL),
#             "api_prefix": str(API_PREFIX),
#             "api_url_prefix": str(API_URL_PREFIX)
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"Failed to check API URL: {str(e)}"
#         }

# @router.get("/evaluate/test-poll-task")
# async def test_poll_task() -> dict[str, str]:
#     """Test endpoint to directly launch poll_and_run_evaluation task."""
#     try:
#         from a4s_eval.celery_tasks import poll_and_run_evaluation

#         # Launch the task directly
#         task = poll_and_run_evaluation.delay()

#         return {
#             "status": "success",
#             "message": "poll_and_run_evaluation task launched directly",
#             "task_id": str(task.id)
#         }
#     except Exception as e:
#         return {
#             "status": "error",
#             "message": f"Failed to launch poll_and_run_evaluation task: {str(e)}"
#         }

# @router.get("/evaluate/test-redis-connectivity")
# async def test_redis_connectivity() -> dict[str, str]:
#     """Test direct Redis connectivity"""
#     import redis
#     import logging
#     from urllib.parse import urlparse
#     logging.basicConfig(level=logging.INFO)
#     logger = logging.getLogger(__name__)

#     try:
#         logger.info("=== TEST REDIS CONNECTIVITY START ===")
#         logger.info(f"1. Redis URL: {env.REDIS_BACKEND_URL}")

#         # Parse Redis URL properly
#         parsed_url = urlparse(env.REDIS_BACKEND_URL)

#         # Extract host and port
#         host = parsed_url.hostname
#         port = parsed_url.port or 6379

#         logger.info(f"2. Connecting to Redis: {host}:{port}")
#         logger.info(f"3. SSL: {parsed_url.scheme == 'rediss'}")

#         # Test direct Redis connection with SSL support
#         if parsed_url.scheme == 'rediss':
#             # SSL Redis connection
#             r = redis.Redis(
#                 host=host,
#                 port=port,
#                 socket_connect_timeout=5,
#                 socket_timeout=5,
#                 ssl=True,
#                 ssl_cert_reqs='none'
#             )
#         else:
#             # Non-SSL Redis connection
#             r = redis.Redis(
#                 host=host,
#                 port=port,
#                 socket_connect_timeout=5,
#                 socket_timeout=5
#             )

#         logger.info("4. Testing Redis ping")
#         ping_result = r.ping()

#         logger.info("5. Testing Redis set/get")
#         r.set('test_key', 'test_value', ex=60)
#         get_result = r.get('test_key')

#         logger.info(f"6. Ping result: {ping_result}")
#         logger.info(f"7. Get result: {get_result}")

#         return {
#             "status": "success",
#             "ping_result": str(ping_result),
#             "get_result": str(get_result),
#             "message": "Redis connectivity test successful"
#         }
#     except Exception as e:
#         logger.error(f"ERROR in test_redis_connectivity: {str(e)}")
#         logger.error(f"Exception type: {type(e)}")
#         import traceback
#         logger.error(f"Traceback: {traceback.format_exc()}")
#         return {
#             "status": "error",
#             "message": f"Redis connectivity test failed: {str(e)}"
#         }
#     finally:
#         logger.info("=== TEST REDIS CONNECTIVITY END ===")
