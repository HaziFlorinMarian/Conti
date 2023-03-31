import random

prime_number_1 = 277
prime_number_2 = 239

ON_low = '0x01'
NOT_low = '0xfe'
''' The plaintext can't be larger than the modulus'''
'''
Euclid's algorithm for determining the greatest common divisor
Use iteration to make it faster for larger integers
'''


def gcd(a, b):
    while b != 0:
        a, b = b, a % b
    return a


'''
Euclid's extended algorithm for finding the multiplicative inverse of two numbers
'''


def multiplicative_inverse(e, phi):
    d = 0
    x1 = 0
    x2 = 1
    y1 = 1
    temp_phi = phi

    while e > 0:
        temp1 = temp_phi // e
        temp2 = temp_phi - temp1 * e
        temp_phi = e
        e = temp2

        x = x2 - temp1 * x1
        y = d - temp1 * y1

        x2 = x1
        x1 = x
        d = y1
        y1 = y

    if temp_phi == 1:
        return d + phi


'''
Tests to see if a number is prime.
'''


def is_prime(num):
    if num == 2:
        return True
    if num < 2 or num % 2 == 0:
        return False
    for n in range(3, int(num ** 0.5) + 2, 2):
        if num % n == 0:
            return False
    return True


def generate_keypair(p, q):
    if not (is_prime(p) and is_prime(q)):
        raise ValueError('Both numbers must be prime.')
    elif p == q:
        raise ValueError('p and q cannot be equal')
    # n = pq
    modulus = p * q

    # Phi is the totient of n
    L = (p - 1) * (q - 1)
    # print("L=",L)

    # Choose an integer e such that e and L(n) are coprime
    e = random.randrange(2, L)
    # print('E=',e)

    # Use Euclid's Algorithm to verify that e and L(n) are comprime
    g = gcd(e, L)
    # print('G=',g)
    while g != 1:
        e = random.randrange(2, L)
        # print("E=",e)
        g = gcd(e, L)
        # print("G=",g)
    # Use Extended Euclid's Algorithm to generate the private key
    d = multiplicative_inverse(e, L)
    # print("D=",d)

    # Return public and private keypair
    # Public key is (e, n) and private key is (d, n)
    return ((e, modulus), (d, modulus))


############################### EXERCISE 1 ###############################
# Complete rsa_library.py with a function called encrypt that encrypts
# the hex number sent by the client, using public key, with the method
# described in the Documentation.
def encrypt(public_key, hex_number):
    e, n = public_key
    plaintext = int(hex_number, 16)
    ciphertext = pow(plaintext, e, n)
    return ciphertext


############################### EXERCISE 2 ###############################
def decrypt(private_key, encrypted_msg):
    d, n = private_key
    plaintext = pow(encrypted_msg, d, n)
    hex_number = "0x" + format(plaintext, '04x')
    return hex_number


############################### EXERCISE 3 ###############################
def low_check(hex_nr):
    low = int(hex_nr[2:4], 16)

    if low == int(ON_low, 16):
        return True
    else:
        return False


############################### EXERCISE 4 ###############################
def number_check(hex_nr):
    high = int(hex_nr[-2:], 16)

    if high == int(NOT_low, 16):
        return True
    else:
        return False
