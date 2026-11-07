use std::fmt::Debug;
use std::ops::{Add, Mul, Sub};

#[derive(Debug)]
struct Vector<K> {
    data: Vec<K>,
    len: usize,
}
#[derive(Debug)]
struct Matrix<K> {
    data: Vec<K>,
    rows: usize,
    cols: usize,
}

impl<K, const N: usize> From<[K; N]> for Vector<K>
where
    K: Copy + Default + Add<Output = K> + Mul<Output = K> + Debug + Sub<Output = K>,
{
    fn from(arr: [K; N]) -> Self {
        Vector {
            data: Vec::from(arr),
            len: N,
        }
    }
}
impl<K, const N: usize, const M:usize> From<[[K; N]; M]> for Matrix<K>
where
    K: Copy + Default + Add<Output = K> + Mul<Output = K> + Debug + Sub<Output = K>,
{
    fn from(arr: [[K; N]; M]) -> Self {
        Matrix {
            data: Vec::from(arr.as_flattened()),
            rows: M,
            cols: N
        }
    }
}
impl<K> std::ops::Add<Vector<K>> for Vector<K>
where
    K: Copy + Default + Add<Output = K> + Mul<Output = K> + Debug + Sub<Output = K>,{
    type Output = Vector<K>;

    fn add(self, rhs: Vector<K>) -> Self::Output {
        let mut r = Vector::new(self.data.len());
        for i in 0..rhs.data.len() {
            r.data[i] =  self.data[i]+ rhs.data[i];
        }
        return self;
    }
}
impl<K> Add<Matrix<K>> for Matrix<K>
where
    K: Copy + Default + Add<Output = K> + Mul<Output = K> + Debug + Sub<Output = K>,{
    type Output = Matrix<K>;

    fn add(self, rhs: Matrix<K>) -> Self::Output {
        assert_eq!(self.cols, rhs.cols);
        assert_eq!(self.rows, rhs.rows);

        let mut r = Matrix::new(self.cols, self.rows);
        for i in 0..rhs.data.len() {
            r.data[i] =  self.data[i]+ rhs.data[i];
        }
        return self;
    }
}


#[allow(dead_code)]
impl <K> Vector<K> where K: Copy + Default + Add<Output = K> + Mul<Output = K> + Debug + Sub<Output = K>
{
    fn new(size:usize) -> Self {
        Vector{data: vec![K::default(); size], len : size}
    }
    fn to_matrix(v: Vector<K>, col:usize, row:usize) -> Option<Matrix<K>> {
        return Matrix::new_from_values(&v.data, col, row)
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
    fn add(mut self, v: Vector<K>) -> () {
        let r = self+v;
        self=r;
    }
    fn sub(&mut self, v: &Vector<K>) -> &mut Self {
        let mut i : usize = 0;
        for e in &mut self.data {
            *e = *e - v.data[i];
            i += 1;
        }
        return self;
    }
    fn scl(&mut self, _a: K) -> () {
        todo!()
    }
}

#[allow(dead_code)]
impl <K>  Matrix<K> where K: Copy + Default + Add<Output = K> + Mul<Output = K> + Debug + Sub<Output = K>
{
    fn new(col:usize, row:usize) -> Self {
        Matrix { data: vec![K::default(); col * row], rows: row, cols: col }
    }
    fn new_from_values(arr:&[K], col:usize, row:usize) -> Option<Self> {
        match arr.len() < col * row  {
            true => None,
            false => Some(Matrix{ data: vec![K::default(); col * row], rows: row, cols: col })
        }
    }
    fn to_vec(m : Matrix<K>) -> Vector<K>{
        Vector{ data:m.data, len:m.rows*m.cols}
    }
    fn size(&self) -> usize {
        return self.data.len();
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
        match row >= self.rows || col >= self.cols {
            true => return None,
            false => {
                return Some(self.data[row * self.cols + col]);
            }
        }
    }
    fn is_square(&self) -> bool {
        return self.cols == self.rows
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
    fn shape(&self) -> (usize, usize) {
        return (self.rows, self.cols);
    }
    fn add(mut self, _v: Matrix<K>) -> () {
        let _res: Matrix<K> = self + _v;
        self = _res;
    }
    fn sub(&mut self, _v: &Matrix<K>) -> () {
        todo!()
    }
    fn scl(&mut self, _a: K) -> () {
        todo!()
    }
}


#[allow(dead_code)]
#[cfg(test)]
mod tests {
    use crate::Matrix;
    use crate::Vector;

    #[test]
    fn creation_m (){
        let m = Matrix::<f32>::from([[2., 3.], [4., 5.]]);
        assert!(m.rows == 2);
        assert!(m.cols == 2);
        assert_eq!(m.get(0, 0).unwrap(), 2.);
        assert_eq!(m.get(0, 1).unwrap(), 3.);
        assert_eq!(m.get(1, 0).unwrap(), 4.);
        assert_eq!(m.get(1, 1).unwrap(), 5.);
    }
    #[test]
    fn is_square_m() {
        let m = Matrix::<f32>::from([[2., 3.], [4., 5.]]);
        assert_eq!(m.is_square(), true);
        let m = Matrix::<f32>::from([[2., 3., 4.], [5., 6., 7.], [8.,9.,10.]]);
        assert_eq!(m.is_square(), true);
        let m = Matrix::<f32>::from([[2.]]);
        assert_eq!(m.is_square(), true);
        let m = Matrix::<f32>::from([[2., 3.], [4., 5.], [6., 7.]]);
        assert_eq!(m.is_square(), false);
        let m = Matrix::<f32>::from([[2., 3., 4.], [5., 6., 7.]]);
        assert_eq!(m.is_square(), false);
        let m = Matrix::<f32>::from([[2., 3.]]);
        assert_eq!(m.is_square(), false);
    }
    #[test]
    fn shape() {
        let m = Matrix::<f32>::from([[2., 3.], [4., 5.]]);
        assert_eq!(m.shape(), (2,2));
        let m = Matrix::<f32>::from([[2., 3.], [4., 5.], [6., 7.]]);
        assert_eq!(m.shape(), (3,2));
        let m = Matrix::<f32>::from([[2., 3., 4.],[5., 6., 7.]]);
        assert_eq!(m.shape(), (2,3));
    }

    // #[test]
    // fn add_m(){
    //     let rhs = Matrix::<f32>::from([[2., 3.], [4., 5.]]);
    //     let mut lhs = Matrix::<f32>::from([[5., 6.], [7., 8.]]);
    //
    //     lhs.dump();
    //     rhs.dump();
    //     lhs.add(rhs);
    //     assert_eq!(lhs.get(1, 1).unwrap(), 5.+8.);
    // }

    #[test]
    fn creation_v(){
        let v = Vector::from([1.,2.,3.,7.,4.]);

        assert!(v.size() == [1.,2.,3.,7.,4.].len());
        assert!(v.get(0).unwrap() == [1.,2.,3.,7.,4.][0]);
        assert!(v.get(1).unwrap() == [1.,2.,3.,7.,4.][1]);
        assert!(v.get(2).unwrap() == [1.,2.,3.,7.,4.][2]);
        assert!(v.get(3).unwrap() == [1.,2.,3.,7.,4.][3]);
        assert!(v.get(4).unwrap() == [1.,2.,3.,7.,4.][4]);
    }

    #[test]
    fn size() {
        let v = Vector::from([2., 3., 4., 5.]);
        assert_eq!(v.size(), 4);
        let v = Vector::from([2., 3., 4., 5., 6., 7.]);
        assert_eq!(v.size(), 6);
    }

}

#[allow(dead_code)]
fn main() {
    println!("cargo test");
    let rhs = Matrix::<f32>::from([[2., 3.], [4., 5.]]);
    let mut lhs = Matrix::<f32>::from([[5., 6.], [7., 8.]]);

    println!("{:?}+{:?}=", lhs, rhs);
    lhs.dump();
    rhs.dump();
    lhs.add(rhs);
    println!("{:?}", lhs);
    assert_eq!(lhs.get(1, 1).unwrap(), 5.+8.);
}
