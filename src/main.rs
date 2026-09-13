use std::fmt::Debug;
use std::ops::{Add, Mul};

struct Vector<K> {
    data: Vec<K>,
    len: usize,
}
struct Matrix<K> {
    data: Vec<K>,
    rows: usize,
    cols: usize,
}

trait VectorAPI<K:Copy + Default + Add + Mul + Debug> {
    fn new(size:usize) -> Self;
    fn new_from_arr(arr:&[K]) -> Self;
    fn to_matrix(v: Vector<K>, col:usize, row:usize) -> Option<Matrix<K>>;
    fn size(&self) -> usize;
    fn set(&mut self, p:usize, value:K) -> bool;
    fn get(&self, p:usize) -> Option<K>;
    fn is_square(&self) -> bool;
    fn dump(&self) -> ();

    fn add(&mut self, v: &Vector<K>) -> ();
    fn sub(&mut self, v: &Vector<K>) -> ();
    fn scl(&mut self, a: K) -> ();
}
trait MatrixAPI<K:Copy + Default + Add + Mul + Debug> {
    fn new(col:usize, row:usize) -> Self;
    fn new_from_arr(arr:&[K], col:usize, row:usize) -> Option<Self> where Self : Sized;
    fn to_vec(m : Matrix<K>) -> Vector<K>;
    fn size(&self) -> usize;
    fn set(&mut self, row:usize, col:usize, value:K) -> bool;
    fn get(&self, row:usize, col:usize) -> Option<K>;
    fn is_square(&self) -> bool;
    fn dump(&self) -> ();

    fn shape(&self) -> (usize, usize);
    fn add(&mut self, v: &Matrix<K>) -> ();
    fn sub(&mut self, v: &Matrix<K>) -> ();
    fn scl(&mut self, a: K) -> ();
}

impl <K:Copy + Default + Add + Mul + Debug> VectorAPI<K> for Vector<K>
{
    fn add(&mut self, v: &Vector<K>) -> () {}
    fn sub(&mut self, v: &Vector<K>) -> () {}
    fn scl(&mut self, a: K) -> () {}
    fn new_from_arr(arr:&[K]) -> Self {
        Vector{data: Vec::from(arr), len : arr.len()}
    }
    fn new(size:usize) -> Self {
        Vector{data: vec![K::default(); size], len : size}
    }
    fn size(&self) -> usize {
        return self.len;
    }
    fn set(&mut self, p:usize, value:K) -> bool {
        match p > self.len {
            true => return false,
            false => {
                self.data[p] = value;
                return true;
            },
        }
    }
    fn get(&self, p:usize) -> Option<K>{
        match p > self.len {
            true => None,
            false=> Some(self.data[p]),
        }
    }
    fn is_square(&self) -> bool {
        let sq = self.len.isqrt();
        return sq * sq == self.len ;
    }
    fn dump(&self) -> () {
        println!("vector:[size:{}, data:{:?}]", self.len, self.data);
    }
    fn to_matrix(v: Vector<K>, col:usize, row:usize) -> Option<Matrix<K>> {
        return Matrix::new_from_arr(&v.data, col, row)
    }
}

impl <K:Copy + Default + Add + Mul + Debug> MatrixAPI<K> for Matrix<K>
{
    fn add(&mut self, v: &Matrix<K>) -> () {}
    fn sub(&mut self, v: &Matrix<K>) -> () {}
    fn scl(&mut self, a: K) -> () {}
    fn new_from_arr(arr:&[K], col:usize, row:usize) -> Option<Self> {
        match arr.len() < col * row  {
            true => None,
            false => Some(Matrix{ data: vec![K::default(); col * row], rows: row, cols: col })
        }
    }
    fn new(col:usize, row:usize) -> Self {
        Matrix { data: vec![K::default(); col * row], rows: row, cols: col }
    }
    fn size(&self) -> usize {
        return self.data.len();
    }
    fn shape(&self) -> (usize, usize) {
        return (self.rows, self.cols);
    }
    fn is_square(&self) -> bool {
        return self.cols == self.rows
    }
    fn set(&mut self, row:usize, col:usize, value:K) -> bool {
        match row > self.rows || col > self.cols {
            true => return false,
            false => {
                self.data[row * self.cols + col] = value;
                return true;
            }
        }
    }
    fn get(&self, row:usize, col:usize) -> Option<K>{
        match row > self.rows || col > self.cols {
            true => return None,
            false => {
                return Some(self.data[row * self.cols + col]);
            }
        }
    }
    fn dump(&self) -> () {
        println!("matrix:[rows:{}, cols:{}, data:", self.rows, self.cols);
        let s:String = self.data.chunks(self.cols).fold(
            String::new(),
            |acc, row| {
                acc + &format!("\t{:?}", row) + "\n"
            }
        );
        println!("{}]", s);
    }
    fn to_vec(m : Matrix<K>) -> Vector<K>{
        return Vector::new_from_arr(&m.data);
    }
}

fn main() {
    let mut v_test:Vector<i32> = Vector::new(5);
    v_test.set(0,1);
    v_test.set(1,2);
    v_test.set(2,3);
    v_test.set(3,4);
    v_test.set(4,5);
    v_test.dump();
    println!("{:?}", v_test.get(1));
    println!("size : {}", v_test.size());
    println!("is square : {}", v_test.is_square());

    let mut m_test:Matrix<i32> = Matrix::new(2,2);
    m_test.set(0,0,1);
    m_test.set(0,1,2);
    m_test.set(1,0,3);
    m_test.set(1,1,4);
    m_test.dump();
    println!("{:?}", m_test.get(0,0));
    println!("size : {}", m_test.size());
    println!("shape : {:?}", m_test.shape());
    println!("is_square : {:?}", m_test.is_square());
}
