-- # compose/mysql/init/init.sql
-- 注意这里的用户名和password必需和docker-compose.yml里与MySQL相关的环境变量保持一致。
GRANT ALL PRIVILEGES ON django_demo.* TO nzb@"%" IDENTIFIED BY "123456";
FLUSH PRIVILEGES;