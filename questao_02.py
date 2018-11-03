#!/usr/local/bin/python
# -*- coding: utf-8 -*-


import numpy as np
import scipy.cluster.hierarchy as shc
import matplotlib.pyplot as plt


class DataGenerator(object):
    def __init__(self):
        self.full_data = np.array(
            [
                [0.88, 0.99, 0.91, 1.10],
                [0.90, 0.99, 0.93, 1.26],
                [0.90, 0.98, 0.94, 1.24],
                [0.87, 0.98, 0.97, 1.18],
                [0.93, 0.93, 0.93, 1.20],
                [0.89, 0.97, 0.92, 1.04],
                [0.88, 0.87, 0.91, 1.41],
                [0.81, 0.92, 0.80, 0.55],
                [0.82, 0.92, 0.75, 1.05],
                [0.85, 0.90, 0.64, 0.07],
                [0.77, 0.85, 0.69, -1.36],
                [0.71, 0.83, 0.72, 0.47],
                [0.75, 0.83, 0.63, -0.87],
                [0.70, 0.62, 0.60, 0.21],
                [0.44, 0.58, 0.37, -1.36],
                [0.47, 0.37, 0.45, -0.68],
                [0.23, 0.33, 0.27, -1.26],
                [0.34, 0.36, 0.51, -1.98],
                [0.31, 0.35, 0.32, -0.55],
                [0.24, 0.37, 0.36, 0.20],
                [0.76, 0.80, 0.61, 0.39],
                # [0.69, 0.75, 0.68, 0.16],
                # [0.24, 0.249, 0.229, 1.056],
            ]
        )
        self.countries = [
            "Reino Unido",
            "Austrália",
            "Canadá",
            "Estados Unidos",
            "Japão",
            "França",
            "Cingapura",
            "Argentina",
            "Uruguai",
            "Cuba",
            "Colômbia",
            "Brasil",
            "Paraguai",
            "Egito",
            "Nigéria",
            "Senegal",
            "Serra Leoa",
            "Angola",
            "Etiópia",
            "Moçambique",
            "China",
            # "Média",
            # "Desvio Padrão",
        ]


class WardLinker(object):
    def __init__(self, data, labels):
        self.data = data
        clusters_0 = [i for i in range(len(data))]
        self.clusters = [clusters_0]
        centers_0 = [np.array(dt) for dt in data]
        self.centers = [centers_0]
        self.last_cluster = max(self.clusters[-1])
        self.groupings = {}
        self.sizes = {i:1 for i in range(len(data))}
        self.labels = labels
        clusters_0 = [i for i in range(len(data))]
        self.all_clusters = [[i] for i in clusters_0]
        self.all_centers = [[i] for i in centers_0]
        self.global_average = np.average(self.centers, axis=0)
        self.r2 = []
        self.t2 = []
    
    def link(self):
        while len(self.clusters[-1]) > 1:
            dists = self.calc_distances()
            idx_a, idx_b, dist = min(dists, key=lambda x: x[2])
            # print(idx_a, idx_b, dist)
            new_clusters = self.clusters[-1]
            new_centers = self.centers[-1]
            cl_b = new_clusters.pop(idx_b)
            cl_a = new_clusters.pop(idx_a)
            ct_b = new_centers.pop(idx_b)
            ct_a = new_centers.pop(idx_a)
            self.last_cluster += 1
            new_clusters.append(self.last_cluster)
            new_centers.append(np.average([ct_a, ct_b], axis=0))
            self.centers.append(new_centers)
            self.clusters.append(new_clusters)
            self.groupings[self.last_cluster] = (cl_a, cl_b)
            self.sizes[self.last_cluster] = self.sizes[cl_a] + self.sizes[cl_b]
            self.all_clusters.append(self.all_clusters[cl_a] + self.all_clusters[cl_b])
            self.all_centers.append(self.all_centers[cl_a] + self.all_centers[cl_b])
            # print(self.clusters)
            # all_average = np.average()
            self.r2.append(self.calc_r2())
            self.t2.append(self.calc_t2(cl_a, cl_b, dist))
        # for i,lbl in enumerate(self.labels):
            # print(i, lbl)
        # for grp in self.groupings:
            # print(grp, self.groupings[grp])
        # print('\nALL CLUSTERS')
        # for dt in self.all_clusters:
        #     print(dt)
        # print('\nALL CENTERS')
        # for dt in self.all_centers:
        #     print(dt)

    def calc_distances(self):
        min_distances = [(0, 0, 1e20) for i in self.clusters[-1]]
        for i, (ct_1, cl_1) in enumerate(zip(self.centers[-1], self.clusters[-1])):
            for j, (ct_2, cl_2) in enumerate(zip(self.centers[-1], self.clusters[-1])):
                if i == j:
                    continue
                dist = np.linalg.norm(ct_1 - ct_2)
                n_a = self.sizes[cl_1]
                n_b = self.sizes[cl_2]
                dist = n_a*n_b/(n_a + n_b) * dist
                if dist < min_distances[i][2]:
                    min_distances[i] = (i, j, dist)
        return min_distances
    
    def calc_r2(self):
        sst = 0
        ssb = 0
        for i, c in enumerate(self.clusters[-1]):
            avg_center = np.average(self.all_centers[c], axis=0)
            for elem_center in self.all_centers[c]:
                sst += np.linalg.norm(avg_center - elem_center)
            ssb += len(self.all_clusters[c]) * np.linalg.norm(self.global_average - avg_center)
        return sst/ssb

    def calc_t2(self, cl_1, cl_2, dist):
        n_1 = len(self.all_clusters[cl_1])
        n_2 = len(self.all_clusters[cl_2])
        if n_1 == 1 and n_2 == 1:
            return 0
        avg_1 = np.average(self.all_centers[cl_1], axis=0)
        avg_2 = np.average(self.all_centers[cl_2], axis=0)
        sum_dist_1 = 0
        sum_dist_2 = 0
        for c in self.all_centers[cl_1]:
            sum_dist_1 += np.linalg.norm(avg_1 - c)
        for c in self.all_centers[cl_2]:
            sum_dist_2 += np.linalg.norm(avg_2 - c)
        print(sum_dist_1, sum_dist_2)
        return dist * (n_1 + n_2 - 2)/(sum_dist_1 + sum_dist_2)


def main():
    generator = DataGenerator()
    # data normalization
    data = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, generator.full_data)

    lnkr = WardLinker(data, generator.countries)
    lnkr.link()

    # print(lnkr.r2)
    ax_1 = plt.subplot(221)
    ax_2 = plt.subplot(222)
    ax_3 = plt.subplot(223)
    ax_4 = plt.subplot(224)
    ax_1.bar([x for x in range(len(lnkr.r2))], lnkr.r2)
    ax_2.bar([x for x in range(len(lnkr.r2))], lnkr.t2)
    ax_3.bar([x for x in range(len(lnkr.r2) - 1)], [b-a for (a,b) in zip(lnkr.r2[:-1], lnkr.r2[1:])], color='orange')
    ax_4.bar([x for x in range(len(lnkr.t2) - 1)], [b-a for (a,b) in zip(lnkr.t2[:-1], lnkr.t2[1:])], color='orange')
    ax_1.set_title('Coeficiente R2') 
    ax_2.set_title('Coeficiente Pseudo T2')
    ax_3.set_title('Variações no coeficiente R2') 
    ax_4.set_title('Varições no coeficiente Pseudo T2') 
    plt.show()

    plt.figure(figsize=(10, 7))  
    plt.title("Customer Dendograms")  
    dend = shc.dendrogram(shc.linkage(data, method='ward'), labels=generator.countries)  
    plt.show()


if __name__ == "__main__":
    main()


# Referências
# http://cda.psych.uiuc.edu/multivariate_fall_2012/systat_cluster_manual.pdf