import logging

class BasicLog:
    @staticmethod
    def get_basic_logger(config):
        logger = logging.getLogger(__name__) 
        logging.basicConfig(
            filename=config.log_dir,
            filemode="a",
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%Y/%m/%d %H:%M:%S",
            level=logging.DEBUG
        )
        return logger