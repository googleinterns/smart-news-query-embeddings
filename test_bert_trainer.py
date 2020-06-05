from bert_trainer import BERTTrainer

if __name__ == '__main__':
    trainer = BERTTrainer('nyt_data_from_2015.pkl')
    preds = trainer.predict([
        "Donald Trump",
        "20th century art",
        "Donald Trump likes 20th century art"
    ])
    print(preds)
