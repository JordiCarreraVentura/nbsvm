import pandas as pd
import re
import sklearn

from sklearn.datasets import fetch_20newsgroups

from sklearn.model_selection import train_test_split


# REGEX_WHITESPACE_NORMALIZATION = re.compile('(\t|\n|\\\\n|\\\\t)', re.IGNORECASE)
# REGEX_BLANKS = re.compile(' {2,}')


def to_csv(rows, path):
    df = pd.DataFrame({
        key: [row[col_id] for row in rows[1:]]
        for col_id, key in enumerate(rows[0])
    })
    df.to_csv(path, sep=';', index=False)


def preprocess_text(text):
#     text = REGEX_WHITESPACE_NORMALIZATION.sub(' ', text)
#     text = REGEX_BLANKS.sub(' ', text)
    return text



class TwentyNewsgroupsWrapper:

    def __init__(
        self,
        randomize=False,
        random_state=1,
        min_chars=50,
        strip_around=0.1
    ):
        self.data = fetch_20newsgroups(
            shuffle=randomize,
            random_state=random_state,
            remove=('headers', 'footers', 'quotes')
        )
        self.min_chars = min_chars
        self.strip_around = strip_around
    
    def __iter__(self):
        for x, y in zip(self.data.data, self.data.target):
            if len(x) < self.min_chars:
                continue
            lines = [line.strip() for line in x.split('\n')]
            if self.strip_around > 0:
                n_stripped_lines = int(len(lines) * self.strip_around)
                lines = lines[n_stripped_lines:-n_stripped_lines]
                text = '\n'.join(lines)
            else:
                text = x
            yield (x, int(y))
    
    def __getitem__(self, doc_id):
        return self.data.data[doc_id]
    
    def split(self, r=0.8, random_sate=1):
        _X, _Y = list(zip(*self))
        X, X_, Y, Y_ = train_test_split(
            _X,
            _Y,
            train_size=r, 
            random_state=random_sate
        )
        return (X, Y), (X_, Y_)
    
    def save(
        self,
        train,
        test,
        path_train='train.tsv',
        path_test='test.tsv'
    ):
        rows_paths = [

            (
                list(zip(
                    [preprocess_text(text) for text in train[0]],
                    train[1]
                )),
                path_train
            ),

            (
                list(zip(
                    [preprocess_text(text) for text in test[0]],
                    test[1]
                )),
                path_test
            )
        ]
        for rows, path in rows_paths:
            to_csv([('text', 'label')] + rows, path)
    



if __name__ == '__main__':
    d = TwentyNewsgroupsWrapper(randomize=True)
    train, test = d.split()
    d.save(train, test)


# dataset = fetch_20newsgroups(shuffle=True, random_state=1, remove=('headers', 'footers', 'quotes'))
# documents = dataset.data
# print(dir(dataset))
# 
# for x, y in zip(dataset.data, dataset.target):
#     print('%s...' % x[:250])
#     print(y)
#     print()