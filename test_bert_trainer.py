from bert_trainer import BERTTrainer

if __name__ == '__main__':
    trainer = BERTTrainer('nyt_data_from_2015.pkl')
    trainer.train_model()
