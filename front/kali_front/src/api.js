import axios from 'axios'

export const BACK_URL = 'http://213.171.28.36:8000'

export const getAllUsers = function (skip, count, filter) {
  return axios
    .get(BACK_URL + '/api/get_users?skip=' + skip + '&count=' + count + '&filter=' + filter)
    .then((resp) => {
      return resp.data
    })
}

export const getAllVacancies = function (skip, count, filter) {
  return axios
    .get(BACK_URL + '/api/get_vacancies?skip=' + skip + '&count=' + count + '&filter=' + filter)
    .then((resp) => {
      return resp.data
    })
}

export const createUser = function (formData) {
  return axios.post(BACK_URL + '/api/create_user', formData).then((resp) => {
    return resp.data
  })
}

export const createVacancy = function (formData) {
  return axios.post(BACK_URL + '/api/create_vacancy', formData).then((resp) => {
    return resp.data
  })
}
