import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset


def process_context_data(users, books, ratings1, ratings2):
    """
    Parameters
    ----------
    users : pd.DataFrame
        users.csv를 인덱싱한 데이터
    books : pd.DataFrame
        books.csv를 인덱싱한 데이터
    ratings1 : pd.DataFrame
        train 데이터의 rating
    ratings2 : pd.DataFrame
        test 데이터의 rating
    ----------
    """

    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left')\
                    .merge(books, on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left')\
                    .merge(books, on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left')\
                    .merge(books, on='isbn', how='left')

    # users 인덱싱 처리
    loc2idx = {v:k for k,v in enumerate(context_df['location'].unique())}
    train_df['location'] = train_df['location'].map(loc2idx)
    test_df['location'] = test_df['location'].map(loc2idx)
    
    age2idx = {v:k for k,v in enumerate(context_df['age_bin'].unique())}
    train_df['age_bin'] = train_df['age_bin'].map(age2idx)
    test_df['age_bin'] = train_df['age_bin'].map(age2idx)

    # book 파트 인덱싱
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}
    train_df['book_author'] = train_df['book_author'].map(author2idx)
    test_df['book_author'] = test_df['book_author'].map(author2idx)
    
    year2idx = {v:k for k,v in enumerate(context_df['year_of_publication'].unique())}
    train_df['year_of_publication'] = train_df['year_of_publication'].map(year2idx)
    test_df['year_of_publication'] = test_df['year_of_publication'].map(year2idx)
    
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    
    # summary 삭제 -> 성능 안나옴
    # train_df.drop(columns=['summary'], inplace=True)
    # test_df.drop(columns=['summary'], inplace=True)
    
    category2idx = {v:k for k,v in enumerate(context_df['major_cat'].unique())}
    train_df['major_cat'] = train_df['major_cat'].map(category2idx)
    test_df['major_cat'] = test_df['major_cat'].map(category2idx)
    
    area2idx = {v:k for k,v in enumerate(context_df['isbn_area'].unique())}
    train_df['isbn_area'] = train_df['isbn_area'].map(area2idx)
    test_df['isbn_area'] = test_df['isbn_area'].map(area2idx)

    idx = {
        "loc2idx":loc2idx,
        "age2idx":age2idx,
        "author2idx":author2idx,
        "year2idx":year2idx,
        "publisher2idx":publisher2idx,
        "category2idx":category2idx,
        "area2idx":area2idx      
    }

    return idx, train_df, test_df


def context_data_load(args):
    """
    Parameters
    ----------
    Args:
        data_path : str
            데이터 경로
    ----------
    """

    ######################## DATA LOAD
    users = pd.read_csv(args.data_path + 'users.csv')
    books = pd.read_csv(args.data_path + 'books.csv')
    train = pd.read_csv(args.data_path + 'train_ratings.csv')
    test = pd.read_csv(args.data_path + 'test_ratings.csv')
    sub = pd.read_csv(args.data_path + 'sample_submission.csv')

    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test)
    
    # user, isbn, location, age_bin, book_author, year_of_publicaiton, publisher, summary, major_cat, isbn_area
    field_dims = np.array([len(user2idx), len(isbn2idx),
                           len(idx['loc2idx']), len(idx['age2idx']), 
                           len(idx['author2idx']), len(idx['year2idx']), len(idx['publisher2idx']), 2,
                           len(idx['category2idx']), len(idx['area2idx']),], dtype=np.uint32)
    print(f"field_dims: {field_dims}")
    print(f"total input dim : {sum(field_dims)}")

    data = {
            'train':context_train,
            'test':context_test.drop(['rating'], axis=1),
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }

    return data


def context_data_split(args, data):
    """
    Parameters
    ----------
    Args:
        test_size : float
            Train/Valid split 비율을 입력합니다.
        seed : int
            랜덤 seed 값
    ----------
    """

    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.test_size,
                                                        random_state=args.seed,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):
    """
    Parameters
    ----------
    Args:
        batch_size : int
            데이터 batch에 사용할 데이터 사이즈
        data_shuffle : bool
            data shuffle 여부
    ----------
    """
    train_dataset = TensorDataset(torch.LongTensor(data['X_train'].values), torch.LongTensor(data['y_train'].values))
    valid_dataset = TensorDataset(torch.LongTensor(data['X_valid'].values), torch.LongTensor(data['y_valid'].values))
    test_dataset = TensorDataset(torch.LongTensor(data['test'].values))

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=args.data_shuffle)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader, valid_dataloader, test_dataloader

    return data
