import logging

TRACE = logging.DEBUG - 5


def setup_logging():
    add_log_level("TRACE", TRACE)
    add_handler()


def set_level(level: int):
    logging.getLogger().setLevel(level)
    handlers = logging.getLogger().handlers
    for h in handlers:
        h.setLevel(level)


def add_handler():
    logging.basicConfig()
    ch = logging.StreamHandler()
    # create formatter and add it to the handlers
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    ch.setFormatter(formatter)
    # add the handlers to the logger
    logging.getLogger().handlers = [ch]


def add_log_level(levelName: str, levelNum: int):
    methodName = levelName.lower()

    def logForLevel(self, message, *args, **kwargs):
        if self.isEnabledFor(levelNum):
            self._log(levelNum, message, args, **kwargs)

    def logToRoot(message, *args, **kwargs):
        logging.log(levelNum, message, *args, **kwargs)

    logging.addLevelName(levelNum, levelName)
    setattr(logging, levelName, levelNum)
    setattr(logging.getLoggerClass(), methodName, logForLevel)
    setattr(logging, methodName, logToRoot)
