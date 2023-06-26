import random
import numpy as np
import cv2
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.backends import default_backend as crypto_default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from numpy.random import randint

from sign import *
import sympy

#1.Initialize the camera and set variables
#2.Allocate square array
#Timing start;
#3.Generation:
#while (NumSoFar < NumNeeded) do
#4.Take one snapshot;  C1
#5.Pick out the brightness ∈ [2, 253];  O(N)
#6.Take the last bits as a SubList;  O(N)
#7.If (Frame is even) ip the bits in SubList;  O(N)
#8.Add SubList to FinalList in row-major order;  C2
#9.NumSoFar = NumSoFar + SubList.Length;  C3
#end while
#Output:
#Print in column-major order;  O(N)
#Extra bits are appended to the tail of the sequence;  C4
#Timing end;

def inRange(value):
    if value > 1 & value < 254:
        return True
    return False

def bits_to_number(list, numberSize):
    number = 0
    for i in range(0,numberSize):
        number += (pow(2,i) * list[1-i])
    return number

#1.Initialize the camera and set variables
video = cv2.VideoCapture('input.mp4')
width = 0
height = 0
numSoFar = 0
numNeeded = 10000
numberSize = 8
frameNumber = 0

#2.Allocate square array
dimension = np.floor(np.sqrt(numNeeded)).astype(int) #The matrix dimension is |_√RequiredLength_|
finalList = np.zeros((dimension,dimension))
print(finalList.shape)

if video.isOpened():
    width = int(video.get(3)) #wyznaczanie szerokosci
    height = int(video.get(4)) #wyznaczanie wysokosci
    #print('width: ', width, ' height: ', height)
    #3.Generation
    while numSoFar < numNeeded:
        frameNumber += 1
        iterator = 1
        #print('numSoFar: ',numSoFar)
        #4.Take snapshot
        ret, frame = video.read()

        ###########################################
        #if ret == True:
            # Display the resulting frame
        #    cv2.imshow('Frame', frame)
            # When everything done, release
            # the video capture object
        #video.release()

        # Closes all the frames
        #cv2.destroyAllWindows()
        ###########################################

        #5.Pick out the brightness ∈ [2, 253]
        subList = []
        for column in range(0,width):
            for row in range(0,height):
                #6.Take the last bits as a SubList
                # BRG = [0,1,2]
                blue = frame[row,column,0]
                if inRange(blue):
                    subList.append(np.bitwise_and(blue,1))  #np.bitwise_and(blue,1) czy tu chodzi o ostatni bit liczby przy pomocy BitAnd?
                    iterator+=1
                red = frame[row,column,1]
                if inRange(red):
                    subList.append(np.bitwise_and(red,1)) #np.bitwise_and(red,1)
                    iterator+=1
                green = frame[row,column,2]
                if inRange(green):
                    subList.append(np.bitwise_and(green,1)) #np.bitwise_and(green,1)
                    iterator+=1
        #7.If (Frame is even) ip the bits in SubList
        if frameNumber%2:
            subList = subList[::-1]
        #8.Add SubList to FinalList in row-major order
        addedNumbers = 0
        for column in range(0,dimension):
            for row in range(0,dimension):
                if addedNumbers < iterator:
                    if finalList[row,column] == 0:
                        if len(subList)!=0:
                            addedNumbers += 1
                            #stringerson = "row" + str(row) + " col " + str(column)
                            #print(stringerson)
                            #cos sie popsuło i raz działa, czasem wychodzi poza zakres tablicy a czasem nie
                            try:
                                finalList[row,column] = subList[addedNumbers]
                            except Exception as e:
                                print()

        # 9.NumSoFar = NumSoFar + SubList.Length
        numSoFar += len(subList)
else:
    print("Failed")

print("Pozyskiwanie szumu zakończone")
print("Post-procesing")
finalList = np.transpose(finalList)
print("Zapisywanie liczb do pliku")

#f = open("zero_one.txt", "w")
#for column in range(0, dimension):
#    for row in range(0, dimension):
#        #finalList[row,column] = (finalList[row,column-1]+finalList[row,column])%2
#        to_write = finalList[row,column].astype(int).astype(str)
#        f.write(to_write)
#        #f.write(' ')
#    #print('\n')
#f.close()

one_dimensional_array = []
for column in range(0, dimension):
    for row in range(0, dimension):
        one_dimensional_array.append(int(finalList[row,column]))


#f = open("random_numbers.txt", "w")
#random_number = 0
#number_counter = 0
#bit8 = np.zeros(numberSize)
#for column in range(0, dimension):
#    for row in range(0, dimension):
#        if number_counter < numberSize:
#            bit8[number_counter] = finalList[row,column]
#            number_counter += 1
#        else:
#            number_counter = 0
#            random_number = bits_to_number(bit8, numberSize)
#            one_dimensional_array.append(random_number)
#            f.write(random_number.astype(str))
#            f.write('\n')
#f.close()

def binary_array_to_decimal(binary_array):
    binary_string = ''.join(str(bit) for bit in binary_array)
    decimal = int(binary_string, 2)
    return decimal

def split_into_2048_bit_numbers(binary_array):
    result = []
    start = 0
    while start + 512 <= len(binary_array):
        number = binary_array[start:start+512]
        decimal = binary_array_to_decimal(number)
        result.append(decimal)
        start += 512
    return result

numbers_to_generator = split_into_2048_bit_numbers(one_dimensional_array)


print("Liczby zostały zapisane")
print("Generowanie p i q")
#========================#
#Koniec Generowania liczb#
#========================#

#========================#
#    Generowanie kluczy  #
#========================#


def gcd(a, b):
    if b > 0:
        return gcd(b, a % b)
    return a

def find_mod_inverse(a, m):
    if gcd(a, m) != 1:
        return None
    u1, u2, u3 = 1, 0, a
    v1, v2, v3 = 0, 1, m

    while v3 != 0:
        q = u3 // v3
        v1, v2, v3, u1, u2, u3 = (u1 - q * v1), (u2 - q * v2), (u3 - q * v3), v1, v2, v3
    return u1 % m


#Nie używane
def isPrimeNumber(number):
    if number < 2:
        return False
    for i in range(2, int(number ** 0.5) + 1):
        if number % i == 0:
            return False
    return True


def liczbyPierwsze():
    p = 0
    q = 0
    flag = False
    for number in numbers_to_generator:
        if sympy.isprime(number):
            flag = True
            p = number
            break

    if flag:
        q = sympy.prevprime(p)
    else:
        random = randint(0, len(numbers_to_generator) - 1)
        x = numbers_to_generator[random]
        p = sympy.prevprime(x)
        q = sympy.prevprime(p)

    return p, q


#def randomNumberFromList(random_prime_number):
#    while(True):
#        random_index = random.randint(1,len(one_dimensional_array)-1)
#        if isPrimeNumber(one_dimensional_array[random_index]) and random_prime_number - 1 > one_dimensional_array[
#            random_index] > 2:
#            return one_dimensional_array[random_index]

def generujKlucze():
    p, q = liczbyPierwsze()
    n = p * q
    euler = (p - 1) * (q - 1)
    e = 3
    while True:
        e = random.randrange(1, euler)
        g = gcd(e, euler)
        if g == 1:
            break

    d = find_mod_inverse(e, euler)
    dmp1 = int(d % (p - 1))
    dmq1 = int(d % (q - 1))
    iqmp = int(pow(q, -1, p))
    public_numbers = rsa.RSAPublicNumbers(e=e, n=n)
    private_key = rsa.RSAPrivateNumbers(
        p=p,
        q=q,
        d=d,
        dmp1=dmp1,
        dmq1=dmq1,
        iqmp=iqmp,
        public_numbers=public_numbers).private_key()

    public_key = public_numbers.public_key()


    return private_key, public_key


def private_key_to_pem(private_key):
    private_key_pem = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return private_key_pem


def public_key_to_pem(public_key):
    public_key_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )

    return public_key_pem


##################################################################################################################
#needed_number = 0
#while(True):
#    random_index = random.randint(1,len(one_dimensional_array)-1)
#    if isPrimeNumber(one_dimensional_array[random_index]):
#        needed_number = one_dimensional_array[random_index]
#        break
private_key, public_key = generujKlucze()
print("Klucze zostały  wygenerowane")

private_key_pem = private_key_to_pem(private_key)
with open("private_key.pem", "wb") as pem_file:
    pem_file.write(private_key_pem)

# Creating PEM form and save public key to file
public_key_pem = public_key_to_pem(public_key)
with open("public_key.pem", "wb") as pem_public_file:
    pem_public_file.write(public_key_pem)

sign()

x = input('Czy można już weryfikować działanie?: ')

with open('public_key.pem', 'rb') as key_file:
    pem_data = key_file.read()

with open('signature.bin', 'rb') as signature_file:
    signature_data = signature_file.read()

with open('file.txt', 'rb') as message_file:
    message = message_file.read()

public_key = serialization.load_pem_public_key(pem_data)
public_key.verify(
    signature_data, #podpis
    message,        #plik do sprawdzenia
    padding.PSS(
        mgf=padding.MGF1(hashes.SHA256()),
        salt_length=padding.PSS.MAX_LENGTH
    ),
    hashes.SHA256()
)