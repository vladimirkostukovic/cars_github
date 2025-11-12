from auto_project.logging_setup import setup_logger

def test_logger_idempotent():
    lg1 = setup_logger("t")
    lg2 = setup_logger("t")
    assert lg1 is lg2
