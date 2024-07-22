

def parse_replica_model_uid(replica_model_uid: str) -> Tuple[str, int, int]:
    """
    Parse replica model uid to model uid, replica and rep id.
    """
    parts = replica_model_uid.split("-")
    if len(parts) == 1:
        return replica_model_uid, -1, -1
    rep_id = int(parts.pop())
    replica = int(parts.pop())
    model_uid = "-".join(parts)
    return model_uid, replica, rep_id