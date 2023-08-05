import axios from 'axios';

export const getAnswer = async (data) => {
    try{
        return await axios.post('http://185.32.161.60:5000/43632', data, {
            headers: {
                "Access-Control-Allow-Credentials": "true", 
                "Content-Type": "application/json"
            }
        });
    }catch(error){
        console.log(error);
    }
}