Чтобы начать процесс, мы загрузим и установим все необходимые элементы из репозиториев Ubuntu. 
pipЧуть позже мы воспользуемся менеджером пакетов Python для установки дополнительных компонентов.

Нам нужно обновить локальный aptиндекс пакетов, а затем загрузить и установить пакеты. 
Пакеты, которые мы устанавливаем, зависят от того, какую версию Python будет использовать ваш проект.

sudo apt-get update
sudo apt-get install python3-pip python3-dev libpq-dev postgresql postgresql-contrib nginx

(.ai_env)pip install gunicorn (if will use the PostgreSQL) - psycopg2
sudo nano /etc/systemd/system/gunicorn.service
sudo systemctl start gunicorn
sudo systemctl enable gunicorn
sudo systemctl status gunicorn

Если вы вносите изменения в /etc/systemd/system/gunicorn.serviceфайл, перезагрузите демон, 
чтобы перечитать определение службы, и перезапустите процесс Gunicorn, набрав:
sudo systemctl daemon-reload
sudo systemctl restart gunicorn