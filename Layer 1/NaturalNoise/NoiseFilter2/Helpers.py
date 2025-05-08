
class Helpers:

    def get_genres(self, dataset):
        genres = []
        items = []

        genres_list = dataset.genres.tolist()
        for genre_category in genres_list:
            g = genre_category.split('|')
            items.extend(g)

        for item in items:
            if item not in genres: 
                genres.append(item)
        
        if '(no genres listed)' in genres_list:
            genres.remove('(no genres listed)')
        
        return genres