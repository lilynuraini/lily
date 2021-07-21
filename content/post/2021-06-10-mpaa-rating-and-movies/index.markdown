---
title: MPAA Rating and Movies
author: Lily Nur Aini
date: '2021-06-10'
slug: mpaa-rating-and-movies
categories: [mpaa and movies]
tags: [mpaa,movie,film,rating,audience]
subtitle: ''
summary: 'What the data tells you about MPAA and movie industry'
authors: []
lastmod: '2021-06-10T13:43:12+07:00'
profile : false
featured: true
draft: false
image:
  caption: '[Image by Krists Luhaers on Unsplash](https://unsplash.com/photos/AtPWnYNDJnM)'
  focal_point: ''
  preview_only: no
projects: []
---







## What's Content Rating?

<img src="Images/mpaa_bump_movie.jpg" width="75%" height="75%" style="display: block; margin: auto;" /><img src="Images/mpaa_bump_trailer.jpg" width="75%" height="75%" style="display: block; margin: auto;" />

You may have seen the above warning cards somewhere for at least once, in movie trailers or before a movie starts. They come in different colors and warnings. This is how the rating association informs you about the content of a movie. The content rating criterias differ for each country but they have the same purpose, to rate the suitability of movies to the audience. Due to these differences and availability of data, the analysis will focus on MPAA content rating and the movies that it rates. MPAA's headquarter is located in the United States so the movies are mostly produced by the movie industry that is often called Hollywood.



## MPAA Rating System

According to the [website](https://www.filmratings.com/), MPAA rating system  was created to help parents make informed viewing choices for their children. To determine the content rating, MPAA raters look for violence, sex, nudity, profanity, and drugs or substance elements in movies then decide the rating codes that suit them. The rating system has changed several times since its establishment in 1922. Now it consists of 5 codes : G, PG, PG-13, R, and NC-17.

![MPAA Poster from [filmratings.com](https://www.filmratings.com/)](Images/mpaa_poster.jpg)

- G : Nothing offensive. Suitable for everybody.

- PG : Some materials may not be suitable for children. Parents need to investigate before letting the younger children watch. The more mature themes may call for parental guidance.

- PG-13 : Some materials may not be appropriate for children under 13. A sterner warning to parents to determine whether their children under age 13 should watch the movie, as some material might not be suited for them.

- R : Some adult materials. Children under 17 are not allowed to watch the movie unaccompanied by a parent or adult guardian. Parents are strongly urged to find out more about the movie in determining their suitability for their children. It is not appropriate for parents to let young children watch R-rated movies.

- NC-17 : Adult materials. Under 18 are not allowed to watch.


## MPAA Rating and Movies

The stick figures from MPAA poster above give us the idea of what kind of audience are appropriate for each rating code. G and PG rating are relatively safe for all as the go-to rating for family movies. Hollywood could take this as an opportunity to make movies that are able to earn G or PG and attract more audience. Afterall, more audience means more money.


<img src="{{< blogdown/postref >}}index_files/figure-html/yearly_chart-1.png" width="672" />

The data proves otherwise. R and PG-13, the MPAA ratings that potentially attract less audience than G and PG, are the top two. R-rated movies dominate the 90's movie production but plummet in the early 2000's. Meanwhile, there is a rise of PG-13 movies that they slowly catch up and take over but only for a short period of time. R continues to dominate and followed by PG-13 in the second place, leaving a big gap for the others.


<img src="{{< blogdown/postref >}}index_files/figure-html/revenue_chart-1.png" width="672" />


The profit data is consistent with MPAA poster. The bigger the audience, the higher the profit. The chart shows that R boxplot has the lowest median (notice the red dash line) compared to the other MPAA ratings. Also, the profit difference between R and other ratings are significant. Comparing profit and number of movie produced, the movie industry needs to make adjustment in their production if they want to maximize their income. [This research](https://www.jstor.org/stable/10.1086/339890) explains more about the topic. The research reports there are too many R-rated movies and suggests Hollywood to relocate the budget to G, PG, and PG-13 movies to cut the risk and increase profit.


From an outsider perspective, Hollywood is willing to take the risk of earning less money in order to keep pumping out R-rated movies. Michael Medved, a veteran movie critic, gives us a little insight on its motivation. In [his book](https://www.amazon.com/Hollywood-America-How-Why-Entertainment/dp/0060924357), he mentions many commercially unsucessful R-rated movies indicate that the audience wants more wholesome entertainment. However, Hollywood ignores them. People in the movie industry have this craving to be accepted and recognized as artists by their peers. Instead of making movies that cater to the masses, they create these movies with mature themes and shocking values to catch the attention, gain the recognition and praised by the critics. These types of movies do appeal to the movie industry elites and they tend to be considered highly artistic.


<img src="{{< blogdown/postref >}}index_files/figure-html/oscar_chart-1.png" width="672" />



One of the parameters for artistic achievement in the movie industry is the award. The Academy Awards, also known as Oscars, is the most prestigious one. There's this common belief that R-rated movies are more likely to win the Oscar. However, when we look closely at the chart above, the compositon shows that R wins just about the same amount as the other ratings. R's higher number of nomination is related to the number of movie released. Since there are more R-rated movies out there, they get more nomination. 



<img src="{{< blogdown/postref >}}index_files/figure-html/metacritic_chart-1.png" width="672" />


Another parameter for achievement is the critics praise. [Metascore](https://www.metacritic.com/about-metascores) is the average score of aggregated reviews given by the leading critics. Movies that get positive reviews get higher Metascore and vice versa. The chart above shows R-rated movies get significantly higher Metascore compared to PG-13 and PG, the two most produced movie ratings after R.


G is another MPAA rating that has significantly higher Metascore than PG and PG-13. The majority of G movies is animated. Box office-wise, they are successful because of the ability to reach wide range of audience. The animation studios sure know how to create compelling stories with low intensity of violence, sex, or profanity but still charm the critics and attract many people. Nowadays, we barely see any G rated movies. Some people argue that G-rated decline is due to stricter MPAA regulation. Let's take Lion King for example. The original 1994 movie was rated [G](https://www.imdb.com/title/tt0110357/?ref_=nv_sr_srsg_0). However, the almost shot-for-shot remake in 2019 got [PG](https://www.imdb.com/title/tt6105098/?ref_=nv_sr_srsg_3). Others argue that [G-rated are considered kids movie and the producers think they don't attract as many audience as they used to](https://www.bostonglobe.com/arts/2013/07/15/rated-movies-fade-black-hollywood-makes-kids-films-more-adult-friendly/oLu8BmdMrC0yEqO4km1dJP/story.html). So they add more mature elements and the movies earn PG rating. 


As for NC-17, although only a few, they get mostly good score from critics. This, once again, proves that the mature contents are more likely to get favorable reviews. However, they're harder to promote and generate small financial return. [Some theater chains refuse to screen them, some media outlets refuse to carry their ads, and some retailers refuse to sell them](https://www.npr.org/2012/08/21/159586654/nc-17-rating-can-be-a-death-sentence-for-movies). That's why many movie creators avoid getting NC-17 from MPAA and aim for R instead. They want to create movies with mature themes, get the praise from peers and critics, but still can reach a fair amount of audience.

<img src="{{< blogdown/postref >}}index_files/figure-html/topblockbuster_chart-1.png" width="672" />

Although the data shows us that Hollywood values acknowledgement more than money, doesn't mean it's not important. Best-selling literature adaptation, sequels, and franchises are strategies for cash grab. Nowadays, these strategies aim for PG-13 rating, the second most produced movie rating, since it's considered still safe for kids, teenagers don't think it's too childish, their adult guardian also can enjoy.

## Note

Data source links

IMDB data
- [IMDB Top 250 Lists and 5000 plus IMDB records](https://data.world/studentoflife/imdb-top-250-lists-and-5000-or-so-data-records)
- [IMDB 5000 Movie Dataset](https://data.world/data-society/imdb-5000-movie-dataset)

Oscar data
- [The Oscar Award, 1927 - 2020](https://www.kaggle.com/unanimad/the-oscar-award)

Movie financial data
- [Opus Data](https://www.opusdata.com/)

Top blockbuster data
- [crowdflower](https://data.world/crowdflower/blockbuster-database)
