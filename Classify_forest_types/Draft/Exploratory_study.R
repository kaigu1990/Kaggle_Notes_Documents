library(dplyr)
library(ggplot2)

data <- read.table(file = "../Desktop/learn-together/train.csv", sep = ",", header = T, stringsAsFactors = F)
names(data)
Hmisc::describe(data)
str(data)
head(Hmisc::describe(data))

data_quan <- data[,2:11]

# Hierarchical clustering
res <- hcut(data_quan, k = 7, stand = TRUE)
fviz_dend(res, rect = TRUE, cex = 0.5)
table(res$cluster)
table(data$Cover_Type)

# Try partitioning methods (K-means), and visualize clustering results
library(factoextra)
library(GGally)
km.res <- kmeans(scale(data_quan), 7, nstart = 25)
fviz_cluster(km.res, data = scale(data_quan),
             #palette = c("#00AFBB","#2E9FDF", "#E7B800", "#FC4E07"),
             ggtheme = theme_minimal(),
             main = "Partitioning Clustering Plot"
)
ggpairs(as.data.frame(scale(data_quan)), columns = 1:9,mapping=aes(colour=as.character(km.res$cluster)))

# PCA
library(FactoMineR)
res.pca <- PCA(data_quan, graph = FALSE)
fviz_pca_contrib(res.pca, choice = "var", axes = 1, top = 10)
fviz_pca_ind(res.pca,
             label = "none", # hide individual labels
             geom.ind = "point",
             mean.point = FALSE,
             invisible = "quali",
             habillage = factor(data$Cover_Type), # color by groups
             #palette = c("#00AFBB", "#E7B800", "#FC4E07"),
             addEllipses = TRUE # Concentration ellipses
)


# Correlation
library(corrplot)
res.cor <- cor(data_quan, method = "spearman")
corrplot(res.cor, type = "lower", 
         addCoef.col = "black", diag = FALSE)

# Density for Hillshade
library(ggplot2)
library(reshape2)
data_hillshade <- data[,c(ncol(data),8:10)]
#data_hillshade <- stack(data_hillshade)
data_hillshade <- melt(data_hillshade, id.vars = "Cover_Type")
names(data_hillshade)[2:3] <- c("Light_type", "Hillshade")
ggplot(data_hillshade, aes(x = Hillshade, fill = Light_type)) +
  geom_density(alpha = 0.5) +
  theme_light()

data_hillshade$Cover_Type <- factor(data_hillshade$Cover_Type)
ggplot(data_hillshade, aes(x = Light_type, y = Hillshade)) +
  geom_boxplot(aes(fill = Cover_Type))



# Boxplot 查看异常值，是否有负数等
library(ggplot2)
library(reshape2)
data_portion2 <- data[,c(ncol(data),2:10)]
data_portion2 <- melt(data_portion2, id.vars = "Cover_Type")
data_portion2$Cover_Type <- factor(data_portion2$Cover_Type)
ggplot(data_portion2, aes(x = variable, y = value)) +
  geom_boxplot(aes(fill = Cover_Type))

ggplot(data_portion3, aes(x = variable, y = value)) +
  geom_boxplot(aes(fill = Cover_Type)) +
  facet_wrap( ~ variable, scales="free")


# Scatter plot
library(ggplot2)
library(GGally)
data_portion3 <- data[,c(ncol(data),2:10)]
data_portion3$Cover_Type <- factor(data_portion3$Cover_Type)
ggpairs(data_portion3, columns=2:10, aes(color=Cover_Type)) + 
  ggtitle("Classify forest types data -- 7 Cover_type")

# Bar Plot
library(ggplot2)
library(reshape2)
data_portion4 <- data[,c(ncol(data),12:15)]
data_portion4$Cover_Type <- factor(data_portion4$Cover_Type)
data_portion4 <- melt(data_portion4, id.vars = "Cover_Type")
ggplot(data_portion4, aes(x = variable, y = value, fill = Cover_Type)) +
  geom_bar(stat = "identity") +
  scale_fill_brewer(palette = "Pastel1")

data_portion5 <- data[,c(ncol(data),16:55)]
data_portion5$Cover_Type <- factor(data_portion5$Cover_Type)
data_portion5 <- melt(data_portion5, id.vars = "Cover_Type")
ggplot(data_portion5, aes(x = variable, y = value, fill = Cover_Type)) +
  geom_bar(stat = "identity") +
  theme_bw() +
  theme(
    axis.text.x = element_text(angle = 60, vjust = 1, hjust = 1)
  ) +
  scale_fill_brewer(palette = "Pastel1")
