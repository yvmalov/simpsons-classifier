# Simpsons classifier

Система построена на базе свёрточной нейронной сети (CNN), за основу взята архитектура Resnet34 обученная на датасете ImageNet. 
На базе этой сети замораживались первые 70% слоёв и обучались последние 30% на датасете: https://www.kaggle.com/ymalov/simpsons. 
Исходный код процесса обучения: https://www.kaggle.com/ymalov/simpsons-baseline

---

### Структура файлов
- Procfile - содержит команды, которые выполняются при запуске приложения
- _init_.py - пустой файл, но обязательный файл
- bot.py - основной код
- index_to_name.json - список имён 42-х персонажей "Симпсонов"
- requirements.txt - список пакетов Python
- resnet34_e30.pth - параметры (веса) модели

---

### Установка бота на хостинг Heroku
- Устанавливаем консольную утилиту
```sh
$ sudo add-apt-repository "deb https://cliassets.heroku.com/branches/stable/apt ./"
$ curl -L https://cli-assets.heroku.com/apt/release.key |
$ sudo apt-key add -
$ sudo apt-get update
$ sudo apt-get install heroku
```
- Заходим на Heroku
```sh
$ heroku login
```
- Создаём новое приложение
```sh
$ heroku create
```
- Добавляем в локальном репозитории удалённый адрес приложения. Вместо `https://git.heroku.com/YOUR_APP.git`, введите адрес удалённого git-репозитория, который получен на предыдущем шаге. 
```sh
$ git remote add heroku https://git.heroku.com/YOUR_APP.git
```
- Отправляем приложение на удалённый сервер Heroku. Это процесс не быстрый и по мере продвижения в консоли появится ход загрузки. Если всё прошло успешно, то ответ должен быть: `Build Success`. Если нет, то система напишет ошибку. Разберитесь в ошибки и отправьте приложение снова.
```sh
$ git push heroku master
```
- Запускаем приложение
```sh
$ heroku ps:scale web=1
```
- Мониторинг работы
```sh
$ heroku logs --tail
```

### Настройка глобальных переменных на Heroku

-- в файле _bot.py_ есть переменные `TOKEN_TG`, `HEROKU_URL`
-- `TOKEN_TG` - токен телеграм-бота, полученный от бота @BotFather
-- `HEROKU_URL` - адрес приложения на Heroku
-- Необходимо перейти на сайт Heroku, зайти в своё приложение с ботом, далее на вкладке _Settings_, в разделе _Config Vars_ добавить эти переменные.

---

### Ссылки
- Бот: https://t.me/mldlcvbot
- Датасет: https://www.kaggle.com/ymalov/simpsons
- Ноутбук с fine-tuning модели Resnet34: https://www.kaggle.com/ymalov/simpsons-baseline
