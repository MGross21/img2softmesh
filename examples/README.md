# Example inputs

| File | Kind | Source | License |
|---|---|---|---|
| `tshirt.jpg` | `tshirt` | [Wikimedia Commons: T-shirt2.jpg](https://commons.wikimedia.org/wiki/File:T-shirt2.jpg) (Elkagye) | public domain |
| `redtee.jpg` | `tshirt` | [Wikimedia Commons: Red-t-shirt-.jpg](https://commons.wikimedia.org/wiki/File:Red-t-shirt-.jpg) | CC BY-SA 4.0 |
| `sock.jpg` | `sock` | [Wikimedia Commons: Argyle-sock.jpg](https://commons.wikimedia.org/wiki/File:Argyle-sock.jpg) | CC BY-SA 2.5 |
| `towel.jpg` | `towel` | [Wikimedia Commons: A sample of cloth for a towel or a tablecloth](https://commons.wikimedia.org/wiki/File:A_sample_of_cloth_for_a_towel_or_a_tablecloth.jpg) | CC BY-SA 4.0 |
| `blanket.jpg` | `blanket` | [Wikimedia Commons: Rio Grande Hispanic Sarape, Cleveland Museum of Art 1921.562](https://commons.wikimedia.org/wiki/File:America,_Native_North_American,_Southwest,_Rio_Grande_Hispanic,_Post-Contac_-_Blanket-_Sarape_-_1921.562_-_Cleveland_Museum_of_Art.jpg) | CC0 |

Images are resized to 1400 px on the long side.

```bash
img2softmesh demo examples/tshirt.jpg -k tshirt --selfcollide --video out/tshirt/demo.mp4
img2softmesh demo examples/tshirt.jpg -k tshirt -g trellis --selfcollide -o out/tshirt_trellis --video out/tshirt_trellis/demo.mp4
img2softmesh demo examples/redtee.jpg -k tshirt --selfcollide --video out/redtee/demo.mp4
img2softmesh demo examples/sock.jpg -k sock --video out/sock/demo.mp4
img2softmesh demo examples/towel.jpg -k towel --video out/towel/demo.mp4
img2softmesh demo examples/blanket.jpg -k blanket --video out/blanket/demo.mp4
```
