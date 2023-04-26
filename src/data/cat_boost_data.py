import pandas as pd
import numpy as np
import os


def load_cat_boost_data(args):
    root_path = os.path.join(os.getcwd(), "../data")

    raw_users = pd.read_csv(f"{root_path}/users.csv")
    raw_books = pd.read_csv(f"{root_path}/books.csv")
    raw_ratings = pd.read_csv(f"{root_path}/train_ratings.csv")

    train_ratings = pd.read_csv(f"{root_path}/train_ratings.csv")
    test_ratings = pd.read_csv(f"{root_path}/test_ratings.csv")

    ### preprocess
    users = users_preprocess(raw_users)
    books, ratings = books_ratings_preprocess(raw_books, raw_ratings)

    ## rating merge
    train_df = train_ratings.merge(users, on="user_id", how="left").merge(
        books[
            [
                "isbn",
                "book_author",
                "year_of_publication",
                "publisher",
                "major_cat",
                "isbn_area",
            ]
        ],
        on="isbn",
        how="left",
    )
    test_df = test_ratings.merge(users, on="user_id", how="left").merge(
        books[
            [
                "isbn",
                "book_author",
                "year_of_publication",
                "publisher",
                "major_cat",
                "isbn_area",
            ]
        ],
        on="isbn",
        how="left",
    )

    train_df["age_bin"] = train_df["age_bin"].astype("str")
    train_df["year_of_publication"] = train_df["year_of_publication"].astype("str")

    test_df["age_bin"] = test_df["age_bin"].astype("str")
    test_df["year_of_publication"] = test_df["year_of_publication"].astype("str")

    train_df["book_author"] = train_df["book_author"].fillna("stephenking")
    test_df["book_author"] = test_df["book_author"].fillna("stephenking")

    return {
        "users": users,
        "books": books,
        "ratings": ratings,
        "train_df": train_df,
        "test_df": test_df,
    }


def users_preprocess(raw_users):
    users = raw_users.copy()

    # location
    users["location"] = (
        users["location"].str.lower().replace("[^0-9a-zA-Z:,]", "", regex=True)
    )
    users["city"] = users["location"].apply(lambda x: x.split(",")[-3].strip())
    users["state"] = users["location"].apply(lambda x: x.split(",")[-2].strip())
    users["country"] = users["location"].apply(lambda x: x.split(",")[-1].strip())
    users = users.replace("na", np.nan)
    users = users.replace("", np.nan)
    users.drop(columns=["location"], inplace=True)

    city_state_map = dict(
        users.groupby("city")["state"].value_counts().sort_values().index.tolist()
    )
    city_country_map = dict(
        users.groupby("city")["country"].value_counts().sort_values().index.tolist()
    )
    users["state"] = users["city"].map(city_state_map)
    users["country"] = users["city"].map(city_country_map)

    # users['location'] = users['country'].copy()
    # users['location'] = np.where(users['location']=='usa',
    #                          users['state'],
    #                          users['location'])
    # users['location'].fillna('na', inplace=True)

    users["country"].fillna("na", inplace=True)
    users["state"].fillna("na", inplace=True)
    users["city"].fillna("na", inplace=True)

    # age
    users["age"].fillna(0, inplace=True)
    bins = [0, 1, 20, 30, 40, 50, 60, 70, 100]
    users["age_bin"] = pd.cut(x=users["age"], bins=bins, right=False, labels=range(8))
    #########
    # 선택
    # location_cnt = users['location'].value_counts()
    # low_cnt_location = location_cnt[location_cnt < 10].index.tolist()
    # for location in low_cnt_location :
    #     users['location'] = np.where(users['location']==location,
    #                                  'others', users['location'])
    ##########
    users.drop(columns=["age"], inplace=True)

    return users


def isbn_area(isbn):
    if isbn[0] in ("0", "1"):
        return "1"
    if isbn[0] in ("2", "3", "4", "5", "7"):
        return isbn[0]
    # 6으로 시작하는 경우 없음
    if isbn[0] == "8":
        return isbn[:2]
    if isbn[0] == "9":
        if int(isbn[:2]) < 95:
            return isbn[:2]
        if int(isbn[:2]) < 99:
            return isbn[:3]
        else:
            return isbn[:4]
    else:
        return "others"


def books_ratings_preprocess(raw_books, raw_ratings):
    books = raw_books.copy()
    ratings = raw_ratings.merge(raw_books[["isbn", "img_url"]], how="left", on="isbn")

    # isbn
    ratings["isbn"] = ratings["img_url"].apply(lambda x: x.split("P/")[1][:10])
    books["isbn"] = books["img_url"].apply(lambda x: x.split("P/")[1][:10])

    # book_author
    books["book_author"] = (
        books["book_author"].str.lower().replace("[^0-9a-zA-Z]", "", regex=True)
    )

    # year_of_publication
    bins = [0, 1950, 1960, 1970, 1980, 1990, 2000, 2010]
    books["year_of_publication"] = pd.cut(
        x=books["year_of_publication"], bins=bins, right=False, labels=range(7)
    )

    # publisher
    books["publisher"] = (
        books["publisher"].str.lower().replace("[^0-9a-zA-Z]", "", regex=True)
    )

    # category
    books["category"] = (
        books["category"].str.lower().replace("[^0-9a-zA-Z]", "", regex=True)
    )
    author_cat_map = dict(
        books.groupby("book_author")["category"]
        .value_counts()
        .sort_values()
        .index.tolist()
    )
    books["category"] = books["book_author"].map(author_cat_map)
    publisher_cat_map = dict(
        books.groupby("publisher")["category"]
        .value_counts()
        .sort_values()
        .index.tolist()
    )
    books["category"] = books["category"].fillna(
        books["publisher"].map(publisher_cat_map)
    )
    books["category"].fillna("na", inplace=True)
    major_cat = [
        "fiction",
        "juvenilefiction",
        "juvenilenonfiction",
        "biography",
        "histor",
        "religio",
        "science",
        "social",
        "politic",
        "humor",
        "spirit",
        "business",
        "cook",
        "health",
        "famil",
        "computer",
        "travel",
        "self",
        "poet",
        "language",
        "art",
        "language art",
        "literary",
        "criticism",
        "nature",
        "philosoph",
        "reference",
        "drama",
        "sport",
        "transportation",
        "comic",
        "craft",
        "education",
        "crime",
        "music",
        "animal",
        "garden",
        "detective",
        "house",
        "tech",
        "photograph",
        "adventure",
        "game",
        "architect",
        "law",
        "antique",
        "friend",
        "sciencefiction",
        "fantasy",
        "mathematic",
        "design",
        "actor",
        "horror",
        "adultery",
    ]
    books["major_cat"] = books["category"].copy()
    for category in major_cat:
        books["major_cat"] = np.where(
            books["category"].str.contains(category), category, books["major_cat"]
        )

    # summary
    books["summary"] = np.where(books["summary"].notnull(), 1, 0)

    # isbn_area
    books["isbn_area"] = books["isbn"].apply(isbn_area)

    #     # 선택
    #     aut_cnt = books['book_author'].value_counts()
    #     low_cnt_aut = aut_cnt[aut_cnt < 10].index.tolist()
    #     for aut in low_cnt_aut :
    #         books['book_author'] = np.where(books['book_author']==aut,
    #                                      'others', books['book_author'])

    #     # 선택
    #     pub_cnt = books['publisher'].value_counts()
    #     low_cnt_pub = pub_cnt[pub_cnt < 10].index.tolist()
    #     for pub in low_cnt_pub :
    #         books['publisher'] = np.where(books['publisher']==pub,
    #                                      'others', books['publisher'])

    #     # 선택
    #     cat_cnt = books['major_cat'].value_counts()
    #     low_cnt_cat = cat_cnt[cat_cnt < 10].index.tolist()
    #     for cat in low_cnt_cat :
    #         books['major_cat'] = np.where(books['major_cat']==cat,
    #                                      'others', books['major_cat'])

    #     # 선택
    #     area_cnt = books['isbn_area'].value_counts()
    #     low_cnt_area = area_cnt[area_cnt < 10].index.tolist()
    #     for area in low_cnt_cat :
    #         books['isbn_area'] = np.where(books['isbn_area']==area,
    #                                      'others', books['isbn_area'])

    ratings.drop(columns=["img_url"], inplace=True)
    books.drop(
        columns=["book_title", "img_url", "language", "category", "img_path"],
        inplace=True,
    )

    return books, ratings
