import math


class PathFileIO:
    def __init__(self, n_hop_bits, ent_id_n_bits, rel_id_n_bits):
        self.n_hop_bits, self.ent_id_n_bits, self.rel_id_n_bits = n_hop_bits, ent_id_n_bits, rel_id_n_bits
        self.MAX_NBIT = 4 
        

    def read_file(self, filepath):
        paths = []
        
        with open(filepath, 'rb') as fp:
            
            while True:
                n_hop_binary = fp.read(self.n_hop_bits)
                n_hop = int.from_bytes(n_hop_binary, byteorder='big')
                if n_hop == 0:
                    break

                nbit = int.from_bytes(fp.read(self.MAX_NBIT), byteorder='big')
                start_ent_id = int.from_bytes(fp.read(nbit), byteorder='big')
                i = 0
                path = [start_ent_id]
                while i < n_hop:
                    nbit = int.from_bytes(fp.read(self.MAX_NBIT), byteorder='big')
                    rel_id = int.from_bytes(fp.read(nbit), byteorder='big')
                    nbit = int.from_bytes(fp.read(self.MAX_NBIT), byteorder='big')
                    ent_id = int.from_bytes(fp.read(nbit), byteorder='big')
                    path.append(rel_id)
                    path.append(ent_id)
                    i += 1
                paths.append(path)
        return paths
                

    def write_to_file(self, path, n_hop, fp):

                fp.write(n_hop.to_bytes(self.n_hop_bits, byteorder='big', signed=False))

                nbit = int(math.log2(path[0]+1))+1
                fp.write(nbit.to_bytes(self.MAX_NBIT, byteorder='big', signed=False)) 
                fp.write(path[0].to_bytes(nbit, byteorder='big', signed=False))
                #fp.write(path[0].to_bytes(self.ent_id_n_bits, byteorder='big', signed=False))
            
                i = 0
                pos = 1
                while i < n_hop:
                    nbit = int(math.log2(path[pos]+1))+1
                    fp.write(nbit.to_bytes(self.MAX_NBIT, byteorder='big', signed=False)) 
                    fp.write(path[pos].to_bytes(nbit, byteorder='big', signed=False))
                    #fp.write(path[pos].to_bytes(self.rel_id_n_bits, byteorder='big', signed=False))
                    nbit = int(math.log2(path[pos+1]+1))+1
                    fp.write(nbit.to_bytes(self.MAX_NBIT, byteorder='big', signed=False)) 
                    fp.write(path[pos+1].to_bytes(nbit, byteorder='big', signed=False))                    
                    #fp.write(path[pos+1].to_bytes(self.ent_id_n_bits, byteorder='big', signed=False))
                    pos += 2
                    i += 1

