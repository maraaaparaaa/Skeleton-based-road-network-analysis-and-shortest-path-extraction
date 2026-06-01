#include "stdafx.h"
#include "common.h"
#include <opencv2/core/utils/logger.hpp>
#include <map>
#include <vector>
#include <queue>

wchar_t* projectPath;

// Vecini
// Ordine circulara: P2(sus), P3(sus-dr), P4(dr), P5(jos-dr),
//                  P6(jos),  P7(jos-st), P8(st), P9(sus-st)
namespace NBH {
    const int SIZE = 8;
    const int di[8] = { -1, -1,  0,  1,  1,  1,  0, -1 };
    const int dj[8] = { 0,  1,  1,  1,  0, -1, -1, -1 };
}

struct Edge {
    int to;
    float cost;
    std::vector<cv::Point> pixels;
};

struct Graph {
    // node_id -> pixel position
    std::vector<cv::Point> vertices;
    // adjacency list
    std::vector<std::vector<Edge>> adj;
    // pixel -> node_id (-1 daca nu e nod)
    std::map<std::pair<int, int>, int> nodeIndex;
};

// Parcurge scheletul de la un nod de start, urmand pixelii albi
// pana ajunge la urmatorul nod; returneaza distanta euclidiana totala
// si nodul destinatie (-1 daca nu gaseste)
float traceEdge(const Mat& sk01,
    cv::Point start, cv::Point prev,
    const std::map<std::pair<int, int>, int>& nodeIndex,
    cv::Point& destination,
    std::vector<cv::Point>& pathPixels)  
{
    float dist = 0.0f;
    cv::Point current = start;
    cv::Point last = prev;

    std::set<std::pair<int, int>> visited;
    int steps = 0;
    int maxSteps = sk01.rows * sk01.cols;

    visited.insert({ prev.y, prev.x });
    pathPixels.clear();
    pathPixels.push_back(prev); // nodul de start

    while (true)
    {
        if (++steps > maxSteps) return -1.0f;

        auto keyCur = std::make_pair(current.y, current.x);
        if (visited.count(keyCur)) return -1.0f;
        visited.insert(keyCur);
        pathPixels.push_back(current);

        if (nodeIndex.count(keyCur))
        {
            destination = current;
            return dist;
        }

        std::vector<cv::Point> candidates;
        for (int k = 0; k < NBH::SIZE; k++)
        {
            int ni = current.y + NBH::di[k];
            int nj = current.x + NBH::dj[k];
            if (ni < 0 || ni >= sk01.rows || nj < 0 || nj >= sk01.cols) continue;
            if (sk01.at<uchar>(ni, nj) != 1) continue;
            auto nkey = std::make_pair(ni, nj);
            if (visited.count(nkey)) continue;
            candidates.push_back(cv::Point(nj, ni));
        }

        if (candidates.empty()) return -1.0f;

        for (auto& c : candidates)
        {
            auto nkey = std::make_pair(c.y, c.x);
            if (nodeIndex.count(nkey))
            {
                float dx = (float)(c.x - current.x);
                float dy = (float)(c.y - current.y);
                dist += sqrtf(dx * dx + dy * dy);
                pathPixels.push_back(c);
                destination = c;
                return dist;
            }
        }

        cv::Point next = candidates[0];
        float dx = (float)(next.x - current.x);
        float dy = (float)(next.y - current.y);
        dist += sqrtf(dx * dx + dy * dy);

        last = current;
        current = next;
    }
}

Graph buildGraph(const Mat& skeleton, const Mat& nodes)
{
    Graph g;

    Mat sk01;
    skeleton.convertTo(sk01, CV_8UC1);
    for (int i = 0; i < sk01.rows; i++)
        for (int j = 0; j < sk01.cols; j++)
            sk01.at<uchar>(i, j) = (sk01.at<uchar>(i, j) > 0) ? 1 : 0;

    // 1. Colectam toti vertexii
    for (int i = 0; i < nodes.rows; i++)
        for (int j = 0; j < nodes.cols; j++)
            if (nodes.at<uchar>(i, j) == 255)
            {
                int id = (int)g.vertices.size();
                g.vertices.push_back(cv::Point(j, i));
                g.nodeIndex[{i, j}] = id;
            }

    int N = (int)g.vertices.size();
    g.adj.resize(N);
    printf("Vertices found: %d\n", N);

    std::set<std::pair<int, int>> addedEdges;

    auto addEdge = [&](int u, int v, float cost, std::vector<cv::Point> pixels) {
        int a = (std::min)(u, v);
        int b = (std::max)(u, v); 
		auto key = std::make_pair(a, b); // seteaza o cheie unica pentru muchie (u,v) indiferent de ordine
		if (!addedEdges.count(key)) { // daca cheia nu e in set, adaugam muchia
            addedEdges.insert(key);
			g.adj[u].push_back({ v, cost, pixels }); // adauga muchia normala cu drumul de la u la v din pixeli
            std::vector<cv::Point> rev(pixels.rbegin(), pixels.rend());
			g.adj[v].push_back({ u, cost, rev }); // adauga muchia inversa cu drumul inversat (pentru vizualizare corecta)
        }
    };

    // 2. Pentru fiecare nod, urmam fiecare ramura
    for (int id = 0; id < N; id++)
    {
		cv::Point v = g.vertices[id]; // pozitia nodului curent

        for (int k = 0; k < NBH::SIZE; k++)
        {
            int ni = v.y + NBH::di[k];
			int nj = v.x + NBH::dj[k]; // pozitia vecinului in directia k
            if (ni < 0 || ni >= sk01.rows || nj < 0 || nj >= sk01.cols) continue;
			if (sk01.at<uchar>(ni, nj) != 1) continue; // daca vecinul e in afara imaginii sau nu e pixel alb, sarim

            // Vecinul e direct un nod?
            auto neighborKey = std::make_pair(ni, nj);
            if (g.nodeIndex.count(neighborKey))
            {
                int otherId = g.nodeIndex[neighborKey];
                float dx = (float)(nj - v.x);
                float dy = (float)(ni - v.y);
                float cost = sqrtf(dx * dx + dy * dy);
                std::vector<cv::Point> pixels = { v, cv::Point(nj, ni) };
				addEdge(id, otherId, cost, pixels); // adauga muchia direct intre cele doua noduri
                continue;
            }

            // Altfel, urmam ramura
            cv::Point dest(-1, -1);
            std::vector<cv::Point> edgePixels;
            float cost = traceEdge(sk01, cv::Point(nj, ni), v, g.nodeIndex, dest, edgePixels);

            if (cost < 0.0f) continue;

            auto destKey = std::make_pair(dest.y, dest.x);
            if (!g.nodeIndex.count(destKey)) continue;

            int destId = g.nodeIndex[destKey];
            addEdge(id, destId, cost, edgePixels); // adauga muchia intre nodul curent si destinatie
        }
    }

    // 3. Print graf
    printf("\n--- Graf ---\n");
    for (int id = 0; id < N; id++)
    {
        printf("V%d (%d,%d) -> ", id, g.vertices[id].x, g.vertices[id].y);
        for (auto& e : g.adj[id])
            printf("V%d (cost=%.2f)  ", e.to, e.cost);
        printf("\n");
    }

    return g;
}


void getNeighbors(const Mat& img, int i, int j, uchar p[8])
{
    for (int k = 0; k < NBH::SIZE; k++)
        p[k] = img.at<uchar>(i + NBH::di[k], j + NBH::dj[k]);
}

// Utilitare skeleton
int countNeighbors(const uchar p[8])
{
    int n = 0;
    for (int k = 0; k < NBH::SIZE; k++) n += p[k];
    return n;
}

int countTransitions(const uchar p[8])
{
    int t = 0;
    for (int k = 0; k < NBH::SIZE; k++)
        t += (p[k] == 0 && p[(k + 1) % NBH::SIZE] == 1);
    return t;
}

// Binarizare 
Mat Gray2Binary(Mat image) {
    int threshold = 127;
    Mat dst = Mat(image.rows, image.cols, CV_8UC1);
    for (int i = 0; i < image.rows; i++)
        for (int j = 0; j < image.cols; j++)
            dst.at<uchar>(i, j) = (image.at<uchar>(i, j) > threshold) ? 255 : 0;
    return dst;
}

// Dilatare 
Mat dilatate(Mat src, int iterations)
{
    Mat result = src.clone();
    for (int it = 0; it < iterations; it++)
    {
        result = src.clone();
        for (int i = 0; i < src.rows; i++)
        {
            for (int j = 0; j < src.cols; j++)
            {
                if (src.at<uchar>(i, j) != 255) continue;
                for (int k = 0; k < NBH::SIZE; k++)
                {
                    int ni = i + NBH::di[k];
                    int nj = j + NBH::dj[k];
                    if (ni >= 0 && ni < src.rows && nj >= 0 && nj < src.cols)
                        result.at<uchar>(ni, nj) = 255;
                }
            }
        }
        src = result.clone();
    }
    return result;
}

// Scheletizare (Zhang-Suen)
Mat computeSkeleton(Mat binary)
{
    Mat img;
    binary.convertTo(img, CV_8UC1);
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
			img.at<uchar>(i, j) = (img.at<uchar>(i, j) > 0) ? 1 : 0;   // pixelii albi devin 1, restul 0

    bool changed = true;
    while (changed)
    {
        changed = false;
		for (int sub = 0; sub < 2; sub++) // cele 2 sub-iteratii
        {
            Mat toDelete = Mat::zeros(img.rows, img.cols, CV_8UC1);
            for (int i = 1; i < img.rows - 1; i++)
            {
                for (int j = 1; j < img.cols - 1; j++)
                {
					if (img.at<uchar>(i, j) != 1) continue; // doar pixelii albi pot fi stersi
                    uchar p[8];
                    getNeighbors(img, i, j, p);

                    int N = countNeighbors(p);
                    int T = countTransitions(p);
                    if (N < 2 || N > 6 || T != 1) continue;

                    // p[0]=P2, p[2]=P4, p[4]=P6, p[6]=P8
                    bool cond;
                    if (sub == 0)
                        cond = (p[0] * p[2] * p[4] == 0) && (p[2] * p[4] * p[6] == 0);
                    else
                        cond = (p[0] * p[2] * p[6] == 0) && (p[0] * p[4] * p[6] == 0);

                    if (cond)
                        toDelete.at<uchar>(i, j) = 1;
                }
            }
            for (int i = 0; i < img.rows; i++)
                for (int j = 0; j < img.cols; j++)
                    if (toDelete.at<uchar>(i, j))
                    {
                        img.at<uchar>(i, j) = 0;
                        changed = true;
                    }
        }
    }

    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            img.at<uchar>(i, j) *= 255;
    return img;
}

Mat detectNodes(Mat skeleton)
{
    int height = skeleton.rows;
    int width = skeleton.cols;
    Mat nodes = Mat::zeros(height, width, CV_8UC1);

    Mat sk01;
    skeleton.convertTo(sk01, CV_8UC1);
    for (int i = 0; i < sk01.rows; i++)
        for (int j = 0; j < sk01.cols; j++)
            sk01.at<uchar>(i, j) = (sk01.at<uchar>(i, j) > 0) ? 1 : 0;

    for (int i = 0; i < height; i++)
    {
        for (int j = 0; j < width; j++)
        {
            if (sk01.at<uchar>(i, j) != 1) continue; // doar pixelii albi pot fi noduri

            // Pixeli pe bordura: nod doar daca e si endpoint sau intersectie
            bool onBorder = (i == 0 || i == height - 1 || j == 0 || j == width - 1);

            uchar p[8] = {};
            // Calculeaza vecini (cu bound check pentru bordura)
            int cnt = 0, trans = 0;
            for (int k = 0; k < NBH::SIZE; k++)
            {
                int ni = i + NBH::di[k];
                int nj = j + NBH::dj[k];
                uchar val = 0;
                if (ni >= 0 && ni < height && nj >= 0 && nj < width)
                    val = sk01.at<uchar>(ni, nj);
                p[k] = val;
            }
            cnt = countNeighbors(p);
            trans = countTransitions(p);

            // Endpoint: 1 vecin (T==1) sau izolat
            // Intersectie: 3+ tranzitii
            if (onBorder)
            {
                // Pe bordura acceptam doar daca e cu adevarat endpoint sau intersectie
                // (nu fiecare pixel de pe o linie dreapta la margine)
                if (cnt == 1 || trans >= 3)
                    nodes.at<uchar>(i, j) = 255;
            }
            else
            {
                if (trans == 1 || trans >= 3)
                    nodes.at<uchar>(i, j) = 255;
            }
        }
    }

    return nodes;
}

// A*
std::vector<int> astar(const Graph& g, int src, int dst)
{
    int N = (int)g.vertices.size();

    std::vector<float> gCost(N, FLT_MAX);
    std::vector<int> prev(N, -1);

    std::priority_queue<std::pair<float, int>,
        std::vector<std::pair<float, int>>,
        std::greater<>> pq;

    // euristica: distanta euclidiana
    auto heuristic = [&](int u) {
        float dx = (float)(g.vertices[u].x - g.vertices[dst].x);
        float dy = (float)(g.vertices[u].y - g.vertices[dst].y);
        return sqrtf(dx * dx + dy * dy);
        };

    gCost[src] = 0.0f;
    pq.push({ heuristic(src), src });

    int iterations = 0;
    int maxIter = 100000; // protectie anti-loop

    while (!pq.empty())
    {
        if (++iterations > maxIter) {
            printf("A* blocat (prea multe iteratii)!\n");
            break;
        }

        auto top = pq.top();
        pq.pop();  

        float f = top.first;
        int u = top.second;

        // daca am ajuns la destinatie
        if (u == dst)
            break;

        // nod depasit (outdated)
        if (f > gCost[u] + heuristic(u))
            continue;

        // relaxare muchii
        for (auto& e : g.adj[u])
        {
            float ng = gCost[u] + e.cost;

            if (ng < gCost[e.to])
            {
                gCost[e.to] = ng;
                prev[e.to] = u;
                pq.push({ ng + heuristic(e.to), e.to });
            }
        }
    }

    // reconstructie drum
    std::vector<int> path;

    if (gCost[dst] == FLT_MAX)
        return path;  // nu exista drum

    for (int cur = dst; cur != -1; cur = prev[cur])
        path.push_back(cur);

    std::reverse(path.begin(), path.end());
    return path;
}

// Vizualizare drum pe imagine
Mat drawPath(const Mat& src, const Graph& g, const std::vector<int>& path)
{
    Mat display;
    cvtColor(src, display, COLOR_GRAY2BGR);

    // Deseneaza toti vertexii
    for (int id = 0; id < (int)g.vertices.size(); id++)
        circle(display, g.vertices[id], 3, Scalar(0, 255, 0), -1);

    if (path.size() < 2) return display;

    // Deseneaza drumul pixel cu pixel pe schelet
    for (int i = 0; i < (int)path.size() - 1; i++)
    {
        int u = path[i], v = path[i + 1];
        for (auto& e : g.adj[u])
        {
            if (e.to == v)
            {
                for (auto& pt : e.pixels)
                    display.at<cv::Vec3b>(pt.y, pt.x) = cv::Vec3b(0, 0, 255);
                break;
            }
        }
    }

    // Marcheaza sursa si destinatia
    circle(display, g.vertices[path.front()], 5, Scalar(255, 0, 0), -1);
    circle(display, g.vertices[path.back()], 5, Scalar(0, 165, 255), -1);

    return display;
}

// Selectare nod prin click
int selectNodeByClick(const std::string& windowName, const Graph& g, const Mat& display)
{
    struct ClickData { cv::Point pt; bool clicked = false; };
    ClickData cd;

    setMouseCallback(windowName, [](int event, int x, int y, int, void* ud) {
        if (event == EVENT_LBUTTONDOWN) {
            auto* d = (ClickData*)ud;
            d->pt = cv::Point(x, y);
            d->clicked = true;
        }
        }, &cd);

    printf("Click pe un nod...\n");
    while (!cd.clicked) waitKey(10);
    setMouseCallback(windowName, nullptr);

    // Gasim nodul cel mai apropiat de click
    int best = -1;
    float bestDist = FLT_MAX;
    for (int id = 0; id < (int)g.vertices.size(); id++)
    {
        float dx = (float)(g.vertices[id].x - cd.pt.x);
        float dy = (float)(g.vertices[id].y - cd.pt.y);
        float d = sqrtf(dx * dx + dy * dy);
        if (d < bestDist) { bestDist = d; best = id; }
    }
    return best;
}

void shortestPath(const Mat& skeleton, const Graph& g)
{
    Mat display;
    cvtColor(skeleton, display, COLOR_GRAY2BGR);
    for (int id = 0; id < (int)g.vertices.size(); id++)
        circle(display, g.vertices[id], 3, Scalar(0, 255, 0), -1);

    imshow("Shortest Path", display);

    printf("\n--- Shortest Path ---\n");
    int src = selectNodeByClick("Shortest Path", g, display);
    printf("Sursa: V%d (%d,%d)\n", src, g.vertices[src].x, g.vertices[src].y);
    circle(display, g.vertices[src], 5, Scalar(255, 0, 0), -1);
    imshow("Shortest Path", display);

    int dst = selectNodeByClick("Shortest Path", g, display);
    printf("Destinatie: V%d (%d,%d)\n", dst, g.vertices[dst].x, g.vertices[dst].y);

    if (src < 0 || dst < 0) {
        printf("Selectie invalida!\n");
        return;
    }

    std::vector<int> path;
    float totalCost = 0.0f;

    path = astar(g, src, dst);
    for (int i = 0; i < (int)path.size() - 1; i++)
        for (auto& e : g.adj[path[i]])
            if (e.to == path[i + 1]) { totalCost += e.cost; break; }

    if (path.empty())
        printf("Nu exista drum intre V%d si V%d!\n", src, dst);
    else
    {
        printf("Drum gasit: ");
        for (int id : path) printf("V%d ", id);
        printf("\nCost total: %.2f px\n", totalCost);

        Mat result = drawPath(skeleton, g, path);

        Mat resized;
        resize(result, resized, Size(), 0.5, 0.5);

        imshow("Shortest Path", resized);
    }

    waitKey(0);
}


// Road Network Analysis
void roadNetworkAnalysis()
{
    char fname[MAX_PATH];
    if (!openFileDlg(fname)) return;

    Mat src = imread(fname, IMREAD_GRAYSCALE);
    if (!src.data) { printf("Error: could not open image.\n"); return; }
    printf("Image loaded: %d x %d\n", src.cols, src.rows);

    Mat binary = Gray2Binary(src);

    Mat skeleton = computeSkeleton(binary);


    imshow("Binary", binary);
    imshow("Skeleton", skeleton);

    Mat nodes = detectNodes(skeleton);
    imshow("Nodes", nodes);

    Graph graph = buildGraph(skeleton, nodes);
    shortestPath(skeleton, graph);

    waitKey(0);
}

int main()
{
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_FATAL);
    projectPath = _wgetcwd(0, 0);
    int op;
    do {
        system("cls");
        destroyAllWindows();
        printf("Menu:\n");
        printf(" 1 - Road Network Analysis\n");
        printf(" 0 - Exit\n\n");
        printf("Option: ");
        scanf("%d", &op);
        switch (op) {
        case 1: roadNetworkAnalysis(); break;
        }
    } while (op != 0);
    return 0;
}