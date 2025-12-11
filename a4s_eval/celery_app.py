from celery import Celery

from a4s_eval.metric_registries import get_n_evaluation
from a4s_eval.utils import env
from a4s_eval.utils.logging import get_logger

logger = get_logger()


logger.debug(f"CELERY REDIS BACKEND URL: {env.REDIS_BACKEND_URL}")
logger.debug("=== CELERY APP INITIALIZATION ===")

celery_app: Celery = Celery(
    __name__, broker=env.CELERY_BROKER_URL, backend=env.REDIS_BACKEND_URL
)

logger.debug("=== CELERY APP CREATED ===")

# Configure Celery settings for 30-minute task execution with heartbeat
celery_config = {
    "task_acks_late": True,  # Acknowledge task only after completion
    "worker_prefetch_multiplier": 1,  # Process one task at a time
    "task_soft_time_limit": 1800,  # 30 minutes soft limit
    "task_time_limit": 2100,  # 35 minutes hard limit (buffer for cleanup)
    "broker_connection_max_retries": 10,
    "redis_retry_on_timeout": True,
    "redis_socket_keepalive": True,
    "redis_socket_keepalive_options": {
        "TCP_KEEPIDLE": 1,
        "TCP_KEEPINTVL": 3,
        "TCP_KEEPCNT": 5,
    },
    "worker_hijack_root_logger": False,
    "broker_use_ssl": env.MQ_USE_SSL,
}

# Only add SSL configuration in production
if env.MQ_USE_SSL and env.BROCKER_SSL_CERT_REQS:
    celery_config["broker_transport_options"] = {
        "ssl": {
            "cert_reqs": 0,  # CERT_NONE as integer
            "ca_certs": None,
            "certfile": None,
            "keyfile": None,
        }
    }

celery_app.conf.update(celery_config)

logger.debug("=== CELERY CONFIGURATION COMPLETED ===")

logger.info(f"{get_n_evaluation()} evaluation(s) registered.")
