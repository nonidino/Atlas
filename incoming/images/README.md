# Drop the website's illustrations here

Upload the generated images to this folder on the branch `atlas-0.1`
(GitHub: open this folder, **Add file → Upload files**, then **Commit changes**).

Name each file by its id and crop, as in the wiki page `website-image-prompts`:

    H1-wide.png    H1-phone.png
    1a-wide.png    1a-phone.png
    2a-wide.png    2a-phone.png
    3a-wide.png    3a-phone.png

Wide is 21:9 (at least 2560 px across), phone is 4:5 (at least 1440 x 1800).
Add a file `generator.txt` saying which tool and version made them, and paste
any prompt you changed. Then tell the chat; it compresses them into `site/images/`
and this folder is emptied. Everything here is public, like the rest of the repository.
