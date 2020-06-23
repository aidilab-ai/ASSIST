
##############################################
MAIN INFOS
##############################################

1) Running dello script per la classificazione dei ticket dell'utente userName
./scripts/classification_script.py userName

2) Running del Training dei modelli per tutti gli utenti
./scripts/training_script.py

3) Creazione di un nuovo utente
./scripts/new_customer.py userName userEmail userPassword customerEmail


##############################################
LOGGING DIR
##############################################
I log degli utenti sono in scripts/log/


##############################################
CREARE UN NUOVO UTENTE MANUALMENTE
##############################################
1-chiamare le API per il signin [https://assist-prj.org/assist/api/v1/signin] passando email,cellphone,password,emailsupport

2-Aggiungere a script/config/config.json un nuovo elemento all'array users ({"name": "USER_NAME", "config_dir": ""})

3-Creare la cartella di configurazione ed i relativi files (category_config, priority_config e connector_config) dell'utente in ticket_classification/config
copiandola dalla cartella base_configs. Cambiare i riferimenti a base_config, che si trovano nei files category_config.json e priority_config.json. Il file connector_config dovrà avere settato
le informazioni riguardanti l'utente che deve connettersi alle API di winet per il login (sono le stesse info utilizzate durante il signin)

4-Copiare e Rinominare la cartella base_model da ~/data/models_and_data/base_model a ~/data/models_and_data/USER_NAME (cambiare anche i riferimenti a base_model nel checkpoint file che si trova in
~/data/models_and_data/USER_NAME/models/category/model/best_models/checkpoint e
~/data/models_and_data/USER_NAME/models/priority/model/best_models/checkpoint )

5-Aggiungere la seguente chiamata al crontab
*/3 * * * * /usr/bin/python3 /home/questit/assist_classifier/scripts/classification_script.py USER_NAME > /home/questit/tmp/classification_log_USER_NAME.txt

