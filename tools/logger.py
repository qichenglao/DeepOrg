import logging
import logging.config


def configure(name, log_path=None):
    if log_path is not None:
        logging.config.dictConfig({
        'version': 1,
        'formatters': {
            # 'complete': {'format': '%(asctime)s - %(name)s - %(funcName)s() - %(levelname)s - %(message)s', 'datefmt': '%Y-%m-%d %H:%M:%S'},
            'default': {'format': '%(asctime)s - %(funcName)s() | %(message)s', 'datefmt': '%Y-%m-%d %H:%M:%S'},
            'msg_only': {'format': '%(message)s'}
        },
        'handlers': {
            'console': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'default',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'default',
                'filename': log_path,
                'maxBytes': 5 * 1024 * 1024,    # 5M
                'backupCount': 3
            },
            'console_msg_only': {
                'level': 'DEBUG',
                'class': 'logging.StreamHandler',
                'formatter': 'msg_only',
                'stream': 'ext://sys.stdout'
            },
            'file_msg_only': {
                'level': 'DEBUG',
                'class': 'logging.handlers.RotatingFileHandler',
                'formatter': 'msg_only',
                'filename': log_path,
                'maxBytes': 5 * 1024 * 1024,    # 5M
                'backupCount': 3
            }
        },
        'loggers': {
            'default': {
                'level': 'DEBUG',
                'handlers': ['console', 'file']
            },
            'msg_only': {
                'level': 'DEBUG',
                'handlers': ['console_msg_only', 'file_msg_only']
            }
        },
        'disable_existing_loggers': False
    })
    else:
        logging.config.dictConfig({
            'version': 1,
            'formatters': {
                'msg_only': {'format': '%(message)s'}
            },
            'handlers': {
                'console_msg_only': {
                    'level': 'DEBUG',
                    'class': 'logging.StreamHandler',
                    'formatter': 'msg_only',
                    'stream': 'ext://sys.stdout'
                },
            },
            'loggers': {
                'console_msg_only': {
                    'level': 'DEBUG',
                    'handlers': ['console_msg_only']
                }
            },
            'disable_existing_loggers': False
        })

    return logging.getLogger(name)


# import logging
# from logging.handlers import RotatingFileHandler
# from logging import handlers
# import sys
#
# log = logging.getLogger('')
# log.setLevel(logging.DEBUG)
# format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#
# ch = logging.StreamHandler(sys.stdout)
# ch.setFormatter(format)
# log.addHandler(ch)
#
# fh = handlers.RotatingFileHandler(LOGFILE, maxBytes=(1048576*5), backupCount=7)
# fh.setFormatter(format)
# log.addHandler(fh)


# import logging
#
# level = logging.INFO
# format = '%(message)s'
# handlers = [logging.FileHandler('filename.log'), logging.StreamHandler()]
# logging.basicConfig(level = level, format = format, handlers = handlers)
#
# logging.info('Hey, this is working!')


# import logging
#
# # set up logging to file - see previous section for more details
# logging.basicConfig(level=logging.DEBUG,
#                     format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
#                     datefmt='%m-%d %H:%M',
#                     filename='myapp.log',
#                     filemode='w')
# # define a Handler which writes INFO messages or higher to the sys.stderr
# console = logging.StreamHandler()
# console.setLevel(logging.INFO)
# # set a format which is simpler for console use
# formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# # tell the handler to use this format
# console.setFormatter(formatter)
# # add the handler to the root logger
# logging.getLogger('').addHandler(console)
#
# # Now, we can log to the root logger, or any other logger. First the root...
# logging.info('Jackdaws love my big sphinx of quartz.')
#
# # Now, define a couple of other loggers which might represent areas in your
# # application:
#
# logger1 = logging.getLogger('myapp.area1')
# logger2 = logging.getLogger('myapp.area2')
#
# logger1.debug('Quick zephyrs blow, vexing daft Jim.')
# logger1.info('How quickly daft jumping zebras vex.')
# logger2.warning('Jail zesty vixen who grabbed pay from quack.')
# logger2.error('The five boxing wizards jump quickly.')
