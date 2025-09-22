# Folder in charge of TMDB API interactions
from client import TMDB_APIClient
from requests.exceptions import HTTPError, Timeout, RequestException


class TMDB_API():
    """Wrapper class for TMDB API interactions"""
    def __init__(self):
        self._client:  TMDB_APIClient = TMDB_APIClient()


    def search_movie(self, title: str):
        """Searches for a movie by title and returns the different potential TMDB ids"""
        try:
            response = self._client.get('search/movie', params={'query': title, 'include_adult': 'true', 'language': 'fr'})
        except Timeout:
            print("Request timed out even with a 30 second timeout.")
        except HTTPError as err:
            print("Received HTTP error:", err)
        except RequestException as err:
            print("Some other error occurred:", err)
        return response

    def get_movie_details(self, movie_id: str):
        """Fetches full movie details from a given movie_id"""
        try:
            response = self._client.get(f'movie/{movie_id}', params={'language': 'fr'})
        except Timeout:
            print("Request timed out even with a 30 second timeout.")
        except HTTPError as err:
            print("Received HTTP error:", err)
        except RequestException as err:
            print("Some other error occurred:", err)
        return response
    
    def find_best_match(self, title: str, known_runtime, top_n = 10, verbose = False, verbose2 = False):
        """
        Searches for a movie by title and finds the best match based on movie duration as a secondary factor.
        """
        
        response = self.search_movie(title)
        num_results = len(response['results'])

        # If no results found give up
        if num_results == 0:
            best_id = find_best_id_decomposed(title, known_runtime)
            if best_id and verbose2:
                print(title)
                print(get_movie_title(best_id))
            return best_id
        

        top_movie_ids = [response["results"][i]['id'] for i in range(num_results)]

        #Search for top_n ids
        if top_n < response["total_results"]:
            top_n_movie_ids = top_movie_ids[:top_n]
        else:
            top_n_movie_ids = top_movie_ids

        # if translated title is the same as the provided title return the corresponding movie_id
        for movie in response["results"][:len(top_n_movie_ids)]:
            n_title = normalize(title)
            n_tmdb_title = normalize(movie["title"])
            if ( (n_title in n_tmdb_title) or (n_tmdb_title in n_title) ):
                best_id = movie['id']
                if verbose:
                    print(get_movie_title(best_id))
                return best_id
            
        # Assuming it is the most unlikely case that the first movie has a runtime of 0 and the match is wrong
        if num_results==1:
            best_id = top_n_movie_ids[0]
            if verbose:
                print(get_movie_title(best_id))
            return best_id 
            
        
        
        # Assuming it is the most unlikely case that the first movie has a runtime of 0 and the match is wrong
        details = get_movie_details(top_n_movie_ids[0])
        if details["runtime"] == 0 or known_runtime == 0:
            best_id = top_n_movie_ids[0]
            if verbose:
                print(get_movie_title(best_id))
            return best_id 
        
        
        top_n_movie_names = [movie["title"] for movie in response["results"][:len(top_n_movie_ids)]]
        if verbose2:
            print(top_n_movie_names)
            print("known_runtime: ", known_runtime)

        # Compare the runtime of the top_n movies with the known runtime and find the best match
        best_id = None
        lowest_diff = float("inf")
        for i, id in enumerate(top_n_movie_ids):
            try:
                details = get_movie_details(id)
            except HTTPError:
                print("unknown id:", id)
                print("unknown title:", top_n_movie_names[i])
            runtime = details["runtime"]
            #print(runtime)
            diff = abs(runtime - known_runtime)
            if diff < lowest_diff:
                lowest_diff = diff
                best_id = id
        if verbose2:
            print(title)
            print(get_movie_title(best_id))
            print("_____________________________________\n")
        return best_id
    
    def find_best_id_decomposed(self, title:str, known_runtime):
        """ If no results are found for the full title, try to decompose the title by common separators """
        
        # try splitting by common separators and searching each part
        if ":" in title:
            left, right = title.split(":", 1)
            for part in (left.strip(), right.strip()):
                best_id = self.find_best_match(part, known_runtime)
                if best_id:
                    return best_id
                
        if "-" in title:
            left, right = title.split("-", 1)
            for part in (left.strip(), right.strip()):
                best_id = self.find_best_match(part, known_runtime)
                if best_id:
                    return best_id

        # give up
        return None