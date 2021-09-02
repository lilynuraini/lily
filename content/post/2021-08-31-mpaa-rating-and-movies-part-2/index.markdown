---
title: MPAA Rating and Movies Part II
author: Lily Nur Aini
date: '2021-08-31'
slug: mpaa-rating-and-movies-part-2
categories: [mpaa and movies]
tags: [mpaa,movie,film,rating,audience,parents,children,movie content,R,Rmarkdown,dplyr]
subtitle: ''
summary: 'MPAA, movie industry, and movie content'
authors: []
lastmod: '2021-08-31T21:18:55+07:00'
profile : false
featured: true
draft: false
image:
  caption: '[Image by Samuel Regan-Asante on Unsplash](https://unsplash.com/photos/wMkaMXTJjlQ)'
  focal_point: ''
  preview_only: no
projects: []
---




**Read part I [here](/post/mpaa-rating-and-movies)**

## MPAA Rating vs Movie's Actual Content

How are MPAA ratings different from each other in terms of movie content and theme? Using the movie plot synopsis data, I generate noun keywords from each MPAA rating from top 50 highest earning movies.

<img src="{{< blogdown/postref >}}index_files/figure-html/top_keyword_chart-1.png" width="960" />

G and PG have similar keywords that are family friendly. PG-13 is in the middle. People see it as a transition between safer contents to more mature ones. Death, weapon, and battle are example of more violent keywords that are on the top list. R and NC-17 are considered more mature ratings. When PG-13 has weapon on its list, R and NC-17 have more specific kind of weapons on the list, gun and chainsaw. The top lists not only contain violent but also sexual theme and mention of drugs.


<img src="{{< blogdown/postref >}}index_files/figure-html/kids_in_mind_chart-1.png" width="672" />


Kids-In-Mind is an independent organization with no affiliation with the movie industry that [provide parents and other adults with objective and complete information about a film’s content so that they can decide, based on their own value system, whether they should watch a movie with or without their kids, or at all](https://kids-in-mind.com/about.htm). According to the [website](https://kids-in-mind.com/about.htm), Kids-In-Mind assign each movie three category ratings: Sex and nudity, violence and gore, and language with the scale of zero to ten, depending on quantity and context.


Using data from Kids-In-Mind, I compare their three category rating score against the MPAA rating. As expected, G and PG are always at the bottom but there are slight increases and G catches up with PG in all categories. G-rated movies nowadays could have the same intensity as PG.


We can also see the increase in PG-13. MPAA often being criticized of its leniency towards violence. One study finds that [PG-13 nowadays is more violent than 1980's R-rated movies](https://www.nbcnews.com/healthmain/pg-13-movies-are-now-more-violent-r-rated-80s-8c11566223). According to [Kids-In-Mind](https://kids-in-mind.com/about.htm), PG-13 is the rating of choice because it has less restrictions, so it's easier to market. Since MPAA has close ties to movie studios and theater chains, the organization has been slowly but surely changing its criteria so that a PG-13 movie today contains far more violence, sexual content and profanity than a few years ago.


While there's a steady rise in PG-13, R's violence stays in relatively the same place. However, sex and language category are more intense than the earlier years. NC-17 only tops in sex category but seems to be in odd positions for violence and language. There are some years when PG-13 is more violent than NC-17.


Overall, the charts show us MPAA rating criteria has changed overtime. It goes to a more intense direction for almost all category in most MPAA rating codes. Meanwhile, [MPAA stands by the rating system](https://www.usnews.com/news/articles/2014/01/07/dont-expect-a-new-movie-ratings-system-in-2014). The organization conducts its own survey involving parents and they think they are getting the correct information from the MPAA rating and they are also happy with the current standard. So how come those parents are confident their children can handle the increasing intensity from movies?


[One research](https://pediatrics.aappublications.org/content/134/5/877.long) suggests that parents may become desensitize towards sex and violence in movies. In this research, parents were shown several violent and sexual scenes from movies then decided the minimum age that are appropriate for viewing. The parents kept lowering their age standard and ended up being more relax towards sex and violence. Daniel Romer, one of the researchers, says [the increase in movie violence and concurrent desensitization may be working to fuel each other](https://www.reuters.com/article/us-movie-violence-kids-idUSKCN0I91WS20141020).


Movies that are considered edgy or shocking years ago, earn lower MPAA rating nowadays for commercial reasons. Moreover, Hollywood keep producing an abundant amount of movies that are full of violent and explicit contents with an inevitable flop for artistic reasons. Sex, violence, and profanity are everywhere that we are accustom to them and we may not even realize there's an undeniable change until it shows on charts.

## Note

kids-in-mind data
- [kids-in-mind website](https://kids-in-mind.com/)

MPST: A Corpus of Movie Plot Synpses with Tags
- [MPST: A Corpus of Movie Plot Synpses with Tags by Sudipta Kar and Suraj Maharjan and A. Pastor López-Monroy and Thamar Solorio](https://ritual.uh.edu/mpst-2018/)
