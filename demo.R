# install.packages("microbenchmark")
library(microbenchmark)

# dyn.load("target/debug/libhottea.so")
dyn.load("target/release/libhottea.so")

ht_add <- function(a, b) {
    .Call("hottea_add", a, b)
}

ht_mean <- function(a) {
    .Call("hottea_mean", a)
}

ht_add(1, 2)

x <- as.numeric(1:100000)


microbenchmark(
    ht_mean(x),
    mean(x)
)


