if __name__ == '__main__':
    import rsa_library

    public_key, private_key = rsa_library.generate_keypair(277, 239)
    print("Public key: ", public_key)
    print("Private key: ", private_key)
    hex_number = '0xfd02'
    encrypted_msg = rsa_library.encrypt(public_key, hex_number)
    print("Encrypted message: ", encrypted_msg)
    decrypted_msg = rsa_library.decrypt(private_key, encrypted_msg)
    print("Decrypted message: ", decrypted_msg)