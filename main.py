from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from bs4 import BeautifulSoup
import sys,requests,re,cv2,os,math
import numpy as  np
import numba
import sqlite3

connection=sqlite3.connect('site.db')
cursor=connection.cursor()
requet=cursor.execute("SELECT lien FROM site").fetchall()
sites=[i[0]for i in requet]

class Window(QMainWindow):
    def __init__(self):
        super().__init__()
        self.resize(1400,900)
        self.setWindowTitle('Image')
        self.setCentralWidget(Widget())

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.search=QLineEdit()
        self.scrollarea=QScrollArea(self)
        self.scrollarea.setWidgetResizable(True)
        self.grid=QGridLayout(self.scrollarea)
        self.data=[]
        self.initUI()

    def initUI(self):
       widgets=[QWidget(),QWidget(),QWidget(),QWidget()]
       layout=[QVBoxLayout(),QVBoxLayout(),QHBoxLayout(),QHBoxLayout()]
       logo=QLabel('Chercher Image')
       logo.setFixedWidth(300)
       logo.setStyleSheet("color:blue")
       logo.setFont(QFont('Arial',20))
       button=QPushButton('Chercher')
       buttonimage=QPushButton('Image')
       button.clicked.connect(self.chercher)
       buttonimage.clicked.connect(self.chercherImage)
       for i in range(2):
           if(i==0):
               layout[1].addWidget(widgets[2])
               layout[1].addWidget(widgets[3])
               layout[2].addWidget(self.search)
               layout[2].addWidget(buttonimage)
               layout[2].addWidget(button)
               layout[3].addWidget(logo)
               widgets[2].setLayout(layout[3])
               widgets[3].setLayout(layout[2])
               widgets[i].setLayout(layout[1])
               layout[i].addWidget(widgets[i],1)
       layout[0].addWidget(self.scrollarea,20)
       widgets[1].setLayout(self.grid)
       self.scrollarea.setWidget(widgets[1])
       self.setLayout(layout[0])

    def chercherImage(self):
        @numba.jit
        def moy(img):
            h, w = img.shape[:2]
            moy = 0
            n = 0
            for i in range(h):
                for j in range(w):
                    moy += img[i, j]
                    n += 1

            return moy / n

        @numba.jit
        def ecart(img, moy):
            h, w = img.shape[:2]
            e = 0.
            n = 0
            for i in range(h):
                for j in range(w):
                    e += (img[i, j] - moy) ** 2
                    n += 1
            e /= n
            return np.sqrt(e)

        def choix_de_fonction(img):
            h, w = img.shape[:2]
            imgCrop = img[0:int(h / 2), 0:int(w / 3)]
            imgCrop1 = img[int(h / 2):h, int(w / 3):2 * int(w / 3)]
            ecar_ty1=ecart(imgCrop, moy(imgCrop))
            ecar_ty2=ecart(imgCrop1, moy(imgCrop1))
            defirent=abs(ecar_ty1-ecar_ty2)
            if defirent<9:
                Res="texture"
            else:
                Res = "forme"
            return Res
        @numba.jit
        def calContras(img):
            temp =0
            h,w = img.shape[:2]
            for j in range(h):
                for i in range(w):
                    temp += img[i,j] * (i-j)**2
            return temp
        @numba.jit
        def calHomogenity(img):
            temp = 0
            h, w = img.shape[:2]
            for j in range(h):
                for i in range(w):
                    temp += img[i, j] / (1+ (i - j) ** 2)
            return temp

        def calEntropy(img):
            temp = 0
            temp -= np.sum(np.multiply(img[img!=0].astype(np.float),np.log10(img[img!=0].astype(np.float))))
            return temp
        @numba.jit
        def calEnergy(img):
            temp = np.sum(np.power(img,2))
            return temp
        def extra_Image(image, taille):
            mat = []
            for i in range(0, taille):
                m = []
                for j in range(0, taille):
                    m.append(image[i, j])
                mat.append(m)
            return np.array(mat)

        def coccurence(img,teta):
            m=extra_Image(img,100)
            temp = np.zeros((255+1,255+1),np.uint8)
            if teta==0:
                starRow = 0
                starCol = 0
                endCol = len(m[0])-1
            elif teta ==45:
                starRow = 1
                starCol = 0
                endCol = len(m[0]) - 1
            elif teta==90:
                starRow = 1
                starCol = 0
                endCol = len(m[0])
            elif teta == 135:
                starRow = 1
                starCol = 1
                endCol = len(m[0])
            for i in range(starRow,len(m)):
                for j in range(starCol,endCol):
                    if teta ==0:
                        temp[m[i,j],m[i,j+1]]+=1
                    elif teta ==45:
                        temp[m[i, j], m[i-1, j + 1]] += 1
                    elif teta ==90:
                        temp[m[i, j], m[i-1, j ]] += 1
                    elif teta ==135:
                        temp[m[i, j], m[i-1, j - 1]] += 1
            total = np.sum(temp)
            temp = temp * 1./(total)
            return calContras(temp),calHomogenity(temp),calEntropy(temp),calEnergy(temp)
        def calCocu(img):
            t1 = img
            Contras1, Homo1, Entropy1, Energy1 = coccurence(t1, 0)
            Contras2, Homo2, Entropy2, Energy2 = coccurence(t1, 45)
            Contras3, Homo3, Entropy3, Energy3 = coccurence(t1, 90)
            Contras4, Homo4, Entropy4, Energy4 = coccurence(t1, 135)
            Contras = (Contras1 + Contras2 + Contras3 + Contras4) / 4
            Homo = (Homo1 + Homo2 + Homo3 + Homo4) / 4
            Entropy = (Entropy1 + Entropy2 + Entropy3 + Entropy4) / 4
            Energy = (Energy1 + Energy2 + Energy3 + Energy4) / 4
            return Contras, Homo, Entropy, Energy
        def Legendepoly(x,p):
            px=0
            for k in range(p+1):
                if (p-k) % 2 == 0:
                    c=((-1)**((p-k)/2))*(x**k)*math.factorial(p+k)/(2**p*math.factorial(k)*math.factorial((p-k)/2)*math.factorial((p+k)/2))
                    px+=c
            return px
        def legendemoments(img,p,q):
            L=0
            for i in range(img.shape[0]):
                for j in range(img.shape[1]):
                    x=(2*i-(img.shape[0]-1))/(img.shape[0]-1)
                    y=(2*j-(img.shape[1]-1))/(img.shape[1]-1)
                    px=Legendepoly(x,p)
                    py=Legendepoly(y,q)
                    x=abs(int(x*(img.shape[0]-1)))
                    y=abs(int(y*(img.shape[1]-1)))
                    L+=img[x][y]*px*py
            return (L*(2*p+1)*(2*q+1))/(img.shape[0]*img.shape[1])

        self.search.setText('')
        while self.grid.count():
            self.grid.takeAt(0).widget().deleteLater()
        fname=QFileDialog().getOpenFileName(self,'Open Image','.\image',"Image files (*.jpg *.gif *.png)")
        self.search.setText(fname[0].split('/')[-1])
        image=cv2.cvtColor(cv2.imread(fname[0]),cv2.COLOR_BGR2GRAY)
        choix=choix_de_fonction(image)
        self.data=[]
        if(choix=="forme"):
             v=[]
             for p in range(3):
                 for q in range(3):
                     v.append(legendemoments(cv2.resize(image,(200,200)),p,q))
             for path in os.listdir('image/form'):
                 d=[]
                 im=cv2.resize(cv2.imread(os.path.join('image/form',path),cv2.IMREAD_GRAYSCALE),(200,200))
                 for p in range(3):
                     for q in range(3):
                         d.append(legendemoments(im,p,q))
                 diference=0
                 for i in range(len(v)):
                     diference+=(v[i]-d[i])**2
                 diference=math.sqrt(diference)
                 if diference < 50:
                    self.data.append(os.path.join('image/form',path))
             self.afficher(self.data)
        else:
            c1,c2,c3,c4=calCocu(image)
            for path in os.listdir('image/texture'):
                t1,t2,t3,t4=calCocu(cv2.imread(os.path.join('image/texture',path),cv2.IMREAD_GRAYSCALE))
                diference = math.sqrt((c1-t1) ** 2 + (c2-t2) ** 2 + (c3-t3) ** 2 + (c4-t4) ** 2)
                if diference < 55:
                    self.data.append(os.path.join('image/texture',path))
        self.afficher(self.data)

    def chercher(self):
        while self.grid.count():
            self.grid.takeAt(0).widget().deleteLater()
        self.data=[]
        dataweb=self.readdataweb()
        for data in dataweb:
            if self.search.text() != '':
                if( self.search.text() in data.find('title').get_text()):
                     for img in data.findAll('img',{'src':re.compile('.jpg')}):
                        self.data.append(img['src'])
                     continue
                else:
                    for img in data.findAll('img',{'src':re.compile('.jpg')}):
                        if self.search.text() in img['src']:
                            self.data.append(img['src'])
                    for img in data.find_all('div', {'class': 'image'}):
                        if self.search.text() in img.find('img')['src']:
                            self.data.append(img.find('img')['src'])
                    txtdata=data.findAll(['p','img'])
                    for i in range(1,len(txtdata)-1) :
                        if(str(txtdata[i]).startswith('<p>')):
                            if self.search.text() in txtdata[i].get_text():
                                if str(txtdata[i-1]).startswith('<img'):
                                    self.data.append(txtdata[i-1]['src'])
                                if str(txtdata[i+1]).startswith('<img'):
                                    self.data.append(txtdata[i+1]['src'])
        self.afficher(self.data)
        print('fin')

    def readdataweb(self):
        datasource=[]
        for site in sites:
            soup=BeautifulSoup(requests.get(site).content,'html.parser')
            datasource.append(soup)
        return datasource

    def afficher(self,data):
       cpt=0
       i=0
       for path in data:
           image=QLabel()
           #label=QLabel(path.split('/')[-1])
           img=QImage()
           if str(path).startswith('//'):
                img.loadFromData(requests.get('http:'+path).content)
           elif str(path).startswith('http:'):
               img.loadFromData(requests.get(path).content)
           else:
               img=QImage(path)
           image.setPixmap(QPixmap(img).scaled(300,300))
           self.grid.addWidget(image,cpt,i)
           #self.grid.addWidget(label,cpt+1,i)
           i+=1
           if i==4:i=0
           if i==0:cpt+=2

if __name__ == '__main__':
    app=QApplication(sys.argv)
    window=Window()
    window.show()
    sys.exit(app.exec_())


