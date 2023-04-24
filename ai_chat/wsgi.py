"""
WSGI config for ai_chat project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.1/howto/deployment/wsgi/
"""

import os,sys
virtenv = os.path.expanduser('~') + '/.ai_env/'
virtualenv = os.path.join(virtenv, 'bin/activate_this.py')
try:
    if sys.version.split(' ')[0].split('.')[0] == '3':
        exec(compile(open(virtualenv, "rb").read(), virtualenv, 'exec'), dict(__file__=virtualenv))
    else:
        execfile(virtualenv, dict(__file__=virtualenv))
except IOError:
    pass
sys.path.append(os.path.expanduser('~'))
sys.path.append(os.path.expanduser('~') + '/ai_chat/')
os.environ['DJANGO_SETTINGS_MODULE'] = 'ai_chat.settings'
from django.core.wsgi import get_wsgi_application
application = get_wsgi_application()

#os.system('celery -A ROOT multi restart worker --pidfile="$HOME/%n.pid" --logfile="$HOME/celery/%n%I.log"')

