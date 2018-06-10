import matplotlib.pyplot as plt
import numpy as np

# add noise to y axis to avoid overlapping
def rand_jitter(arr):

	# "Range" = (max(arr) - min(arr))
    nosie = .01 * (max(arr) - min(arr))
	
    return arr + np.random.randn(len(arr))



# https://stackoverflow.com/questions/4383571/importing-files-from-different-folder
# # some_file.py
# import sys
# sys.path.insert(0, '/path/to/application/app/folder')
# import some_file
# ...
# sys.path.append('/path/to/application/app/folder') is cleaner imo
# ...
# Yep, it is, but inserting it at the beginning has the benefit of guaranteeing that the path is searched before others (even built-in ones) in the case of naming conflicts.
# https://docs.python.org/2/tutorial/modules.html#packages



# https://stackoverflow.com/questions/17547699/what-does-the-jitter-function-do-in-r
# According to the documentation, the explanation for the jitter function is "Add a small amount of noise to a numeric vector."
# ...
# Jittering indeed means just adding random noise to a vector of numeric values, by default this is done in jitter-function by drawing samples from the uniform distribution. The range of values in the jittering is chosen according to the data, if amount-parameter is not provided.
# I think term 'jittering' covers other distributions than uniform, and it is typically used to better visualize overlapping values, such as integer covariates.
# This helps grasp where the density of observations is high.
# It is good practice to mention in the figure legend if some of the values have been jittered, even if it is obvious.
# Here is an example visualization with the jitter-function as well as a normal distribution jittering where I arbitrarily threw in value sd=0.1
# ...
# A really good explanation of the Jitter effect and why it is necessary can be found in the Swirl course on Regression Models in R.
# The course says that if you do not have jitter, many people will have the same height, so points falls on top of each other which is why some of the circles in the first plot look darker than others. However, by using R's function "jitter" on the children's heights, we can spread out the data to simulate the measurement errors and make high frequency heights more visible.
# http://swirlstats.com/scn/regmod.html
# https://github.com/swirldev/swirl_courses/tree/master/Regression_Models


# https://thomasleeper.com/Rcourse/Tutorials/jitter.html



# http://stat.ethz.ch/R-manual/R-devel/library/base/html/jitter.html
# Description
#   Add a small amount of noise to a numeric vector.
# Usage
#   jitter(x, factor = 1, amount = NULL)
# Arguments
#   x 	
#     numeric vector to which jitter should be added.
#   factor 	
#     numeric.
#   amount 	
#     numeric; if positive, used as amount (see below), otherwise, if = 0 the default is factor * z/50.
#     Default (NULL): factor * d/5 where d is about the smallest difference between x values.
# Details
#   The result, say r, is r <- x + runif(n, -a, a) where n <- length(x) and a is the amount argument (if specified).
#   Let z <- max(x) - min(x) (assuming the usual case). The amount a to be added is either provided as positive argument amount or otherwise computed from z, as follows:
#   If amount == 0, we set a <- factor * z/50 (same as S).
#   If amount is NULL (default), we set a <- factor * d/5 where d is the smallest difference between adjacent unique (apart from fuzz) x values.
# Value
#   jitter(x, ...) returns a numeric of the same length as x, but with an amount of noise added in order to break ties. 
# See Also: http://stat.ethz.ch/R-manual/R-devel/library/graphics/html/rug.html

#> jitter
#function (x, factor = 1, amount = NULL) 
#{
#    if (length(x) == 0L) 
#        return(x)
#    if (!is.numeric(x)) 
#        stop("'x' must be numeric")
#    z <- diff(r <- range(x[is.finite(x)]))
#    if (z == 0) 
#        z <- abs(r[1L])
#    if (z == 0) 
#        z <- 1
#    if (is.null(amount)) {
#        d <- diff(xx <- unique(sort.int(round(x, 3 - floor(log10(z))))))
#        d <- if (length(d)) 
#            min(d)
#        else if (xx != 0) 
#            xx/10
#        else z/10
#        amount <- factor/5 * abs(d)
#    }
#    else if (amount == 0) 
#        amount <- factor * (z/50)
#    x + stats::runif(length(x), -amount, amount)
#}
#<bytecode: 0x000000001d198300>
#<environment: namespace:base>
