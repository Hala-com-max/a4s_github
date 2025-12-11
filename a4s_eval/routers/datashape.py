import uuid
from fastapi import APIRouter

from a4s_eval.tasks.datashape_tasks import auto_discover_datashape

router = APIRouter(tags=["datashapes"])


@router.get("/datashape/autodiscover/{datashape_pid}")
async def autodiscover(
    datashape_pid: uuid.UUID,
) -> dict[str, str]:
    auto_discover_datashape.delay(datashape_pid)
    return {"message": "discovery started."}
