<template>
  <form enctype="multipart/form-data" @submit.prevent="createUserHandler">
    <div class="mb-3 row">
      <label for="name" class="col-sm-2 col-form-label">Имя</label>
      <div class="col-sm-10">
        <input type="text" class="form-control" id="name" name="name" />
      </div>
    </div>
    <div class="mb-3 row">
      <label for="video" class="form-label">Визитка</label>
      <input class="form-control" type="file" id="video" name="video" />
    </div>
    <div class="mb-3 row">
      <label for="transcription" class="form-label">Транскрипция</label>
      <textarea class="form-control" id="transcription" rows="3" name="transcription"></textarea>
    </div>
    <button type="submit" class="btn btn-primary">Обработать</button>
  </form>
</template>

<script>
import { createUser } from '@/api'

export default {
  emits: {
    user_created: Object,
  },
  methods: {
    async createUserHandler(event) {
      let formData = new FormData(event.target)
      let response = await createUser(formData)
      this.$emit('user_created', response)
    },
  },
}
</script>

<style></style>
